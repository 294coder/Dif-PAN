import argparse
import os
import os.path as osp

import h5py
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as data
import time
import colored_traceback.always
from wandb.util import generate_id

from datasets.FLIR_2 import FLIRDataset
from datasets.HISR import HISRDatasets
from datasets.TNO import TNODataset
from datasets.gf import GF2Datasets
from datasets.wv3 import WV3Datasets, make_datasets
from engine import train
from model import build_network
from utils import (
    TensorboardLogger,
    TrainProcessTracker,
    config_load,
    convert_config_dict,
    get_optimizer,
    get_scheduler,
    h5py_to_dict,
    is_main_process,
    merge_args_namespace,
    module_load,
    resume_load,
    BestMetricSaveChecker,
    set_all_seed,
    get_loss,
    get_fusion_dataset
)


def get_args():
    parser = argparse.ArgumentParser("PANFormer")

    # network
    parser.add_argument("-a", "--arch", type=str, default="pannet")
    parser.add_argument("--sub_arch", default=None, help="panformer sub-architecture name")

    # train config
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_id", type=str, default=None)
    parser.add_argument("--non_load_strict", action="store_false", default=True)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("--val_n_epoch", type=int, default=30)
    parser.add_argument("--warm_up_epochs", type=int, default=80)
    parser.add_argument( "-l", "--loss", type=str, default="mse", choices=[ "mse", "l1", "hybrid", "smoothl1", "l1ssim", "charbssim", "ssimsf", "ssimmci", "mcgmci", "ssimrmi_fuse", "pia_fuse", "u2fusion", "swinfusion", "hpm", "none", "None",],)
    parser.add_argument("--grad_accum_ep", type=int, default=None)
    parser.add_argument("--save_every_eval", action="store_true", default=False)

    # resume training config
    parser.add_argument("--resume_ep", default=None, required=False, help="do not specify it")
    parser.add_argument("--resume_lr", type=float, required=False, default=None)
    parser.add_argument("--resume_total_epochs", type=int, required=False, default=None)

    # path and load
    parser.add_argument("-p", "--path", type=str, default=None, help="only for unsplitted dataset")
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--load", action="store_true", default=False, help="resume training")
    parser.add_argument("--save_base_path", type=str, default="./weight")

    # datasets config
    parser.add_argument("--dataset", type=str, default="wv3")
    parser.add_argument("-b", "--batch_size", type=int, default=1028)
    parser.add_argument("--hp", action="store_true", default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--aug_probs", nargs="+", type=float, default=[0.0, 0.0])
    parser.add_argument("-s", "--seed", type=int, default=3407)
    parser.add_argument("-n", "--num_worker", type=int, default=8)
    parser.add_argument("--ergas_ratio", type=int, choices=[2, 4, 8, 16, 20], default=4)

    # logger config
    parser.add_argument("--logger_on", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="panformer_wv3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument( "--resume", type=str, default="None", help="used in wandb logger, please not use it in tensorboard logger")
    parser.add_argument("--run_id", type=str, default=generate_id())
    parser.add_argument("--watch_log_freq", type=int, default=10)
    parser.add_argument("--watch_type", type=str, default="None")
    parser.add_argument("--log_metrics", action="store_true", default=False)

    # ddp setting
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true", default=False)

    # some comments
    parser.add_argument("--comment", type=str, required=False, default="")

    return parser.parse_args()


def main(local_rank, args):
    if type(local_rank) == str: 
        local_rank = int(local_rank.split(':')[-1])
    set_all_seed(args.seed + local_rank)
    torch.cuda.set_device(local_rank if args.ddp else args.device)

    # load other configuration and merge args and configs together
    configs = config_load(args.arch, "./configs")
    args = merge_args_namespace(args, convert_config_dict(configs))
    if is_main_process():
        print(args)

    # define network
    full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
    args.full_arch = full_arch
    network_configs = getattr(
        args.network_configs, full_arch, args.network_configs
    ).to_dict()
    network = build_network(full_arch, **network_configs).cuda()
    # FIXME: may raise compile error
    # network = torch.compile(network)

    # parallel or not
    assert not (args.dp and args.ddp), "dp and ddp can not be True at the same time"
    if args.dp:
        network = nn.DataParallel(network, list(range(torch.cuda.device_count())), 0)
    elif args.ddp:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=torch.cuda.device_count(),
            rank=local_rank,
        )
        network = network.to(local_rank)
        args.optimizer.lr *= args.world_size
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = nn.parallel.DistributedDataParallel(
            network,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True,
        )
    else:
        network = network.to(args.device)

    # optimization
    if args.grad_accum_ep is not None:
        lr_adjust_ratio = args.grad_accum_ep
    else:
        lr_adjust_ratio = 1.0
    args.optimizer.lr *= lr_adjust_ratio
    optim = get_optimizer(network.parameters(), **args.optimizer.to_dict())
    lr_scheduler = get_scheduler(optim, **args.lr_scheduler.to_dict())
    criterion = get_loss(args.loss, network_configs.get("spectral_num", 4)).cuda()

    # load params
    # assert not (args.load and args.resume == 'allow'), 'resume the network and wandb logger'
    status_tracker = TrainProcessTracker(id=args.run_id, resume=args.load, args=args)

    # pretrain load
    if args.pretrain:
        assert (
            args.pretrain_id is not None
        ), "you should specify @pretrain_id when @pretrain is True"
        p = osp.join(args.save_base_path, args.arch + "_" + args.pretrain_id + ".pth")
        network = module_load(
            p,
            network,
            args.device,
            local_rank if args.ddp else None,
            strict=args.non_load_strict,
        )
        if is_main_process():
            print("*" * 20, f"load pretrain weight id: {args.pretrain_id}", "*" * 20)

    # resume training
    if args.load:
        args.resume = "allow"
        p = osp.join(
            args.save_base_path, args.arch + "_" + status_tracker.status["id"] + ".pth"
        )
        if args.resume_total_epochs is not None:
            assert (
                args.resume_total_epochs == args.epochs
            ), "@resume_total_epochs should equal to @epochs"
        network, optim, lr_scheduler, args.resume_ep = resume_load(
            p,
            network,
            optim,
            lr_scheduler,
            device=args.device,
            specific_resume_lr=args.resume_lr,
            specific_epochs=args.resume_total_epochs,
            ddp_rank=local_rank if args.ddp else None,
            ddp=args.ddp,
        )
        if is_main_process():
            print(f"load {p}", f"resume from epoch {args.resume_ep}", sep="\n")

    # get logger
    if is_main_process() and args.logger_on:
        # logger = WandbLogger(args.proj_name, config=args, resume=args.resume,
        #                      id=args.run_id if not args.load else status_tracker.status['id'], run_name=args.run_name)
        logger = TensorboardLogger(
            comment=args.run_id,
            args=args,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
        logger.watch(
            network=network.module if args.ddp else network,
            watch_type=args.watch_type,
            freq=args.watch_log_freq,
        )
    else:
        from utils import NoneLogger
        logger = NoneLogger()

    # get datasets and dataloader
    train_ds, val_ds = get_fusion_dataset(args)
        
    # from torch.utils.data import Subset
    # train_ds = Subset(train_ds, range(0, 20))
        
    if args.ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=args.shuffle)
        val_sampler = torch.utils.data.DistributedSampler(val_ds, shuffle=args.shuffle)
    else:
        train_sampler, val_sampler = None, None
    
    train_dl = data.DataLoader(
        train_ds,
        args.batch_size,
        num_workers=args.num_worker,
        sampler=train_sampler,
        prefetch_factor=8,
        pin_memory=True,
        shuffle=args.shuffle if not args.ddp else None,
    )
    val_dl = data.DataLoader(
        val_ds,
        1,
        # args.batch_size,
        num_workers=args.num_worker,
        sampler=val_sampler,
        pin_memory=True,
        shuffle=args.shuffle if not args.ddp else None,
    )

    # make save path legal
    if not args.save_every_eval:
        args.save_path = osp.join(
            args.save_base_path, args.arch + "_" + args.run_id + ".pth"
        )
        if not osp.exists(args.save_base_path):
            os.makedirs(args.save_base_path)
    else:
        args.save_path = osp.join(args.save_base_path, args.arch + "_" + args.run_id)
        if not osp.exists(args.save_path):
            os.makedirs(args.save_path)
    print("network params are saved at {}".format(args.save_path))

    # save checker
    save_checker = BestMetricSaveChecker(metric_name="PSNR", check_order="up")

    # start training 
    # TODO: use safetensor
    with status_tracker:
        train(
            network,
            optim,
            criterion,
            args.warm_up_epochs,
            lr_scheduler,
            train_dl,
            val_dl,
            args.epochs,
            args.val_n_epoch,
            args.save_path,
            logger=logger,
            resume_epochs=args.resume_ep or 1,
            ddp=args.ddp,
            check_save_fn=save_checker,
            fp16=args.fp16,
            max_norm=args.max_norm,
            grad_accum_ep=args.grad_accum_ep,
            args=args,
        )

    # logger finish
    status_tracker.update_train_status("done")
    if is_main_process() and logger is not None:
        # logger.run.finish()
        logger.writer.close()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5700"

    args = get_args()
    # print(args)

    if (not args.ddp) and (not args.dp): 
        print('using one gpu')
        main(args.device, args)
    else:
        print('SPAWN: using multiple gpus')
        # TODO: resort to accelerator method, to avoid multi-processing dataloader using more RAM
        mp.spawn(main, args=(args,), nprocs=args.world_size if args.ddp else 1)
