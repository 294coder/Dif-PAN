import os
from functools import partial
import math
from typing import Callable, List, Union

import einops
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import nullcontext

from model.base_model import BaseModel
from utils import (
    AnalysisPanAcc,
    AnalysisFLIRAcc,
    LinearWarmupScheduler,
    WandbLogger,
    dict_to_str,
    is_main_process,
    prefixed_dict_key,
    res_image,
    step_loss_backward,
    accum_loss_dict,
    ave_ep_loss,
    ep_loss_dict2str,
    ave_multi_rank_dict,
    NonAnalysis,
)
from utils.log_utils import TensorboardLogger


@torch.no_grad()
def val(
        network: BaseModel,
        val_dl: DataLoader,
        criterion: Callable,
        logger: Union[WandbLogger, TensorboardLogger],
        ep: int = None,
        optim_val_loss: float = None,
        args=None,
):
    val_loss = 0.0
    i = 0

    if args.log_metrics:
        if args.dataset in ["wv3", "qb", "gf", "hisi"]:
            analysis = AnalysisPanAcc(args.ergas_ratio)
        else:
            analysis = AnalysisFLIRAcc()
    else:
        analysis = NonAnalysis()
    val_loss_dict = {}
    for i, (pan, ms, lms, gt) in enumerate(val_dl, 1):
        pan = pan.cuda().float()
        ms = ms.cuda().float()
        lms = lms.cuda().float()
        gt = gt.cuda().float()

        sr = network(ms, lms, pan, mode="eval")

        loss_out = criterion(sr, gt)
        # if loss is hybrid, will return tensor loss and a dict
        if isinstance(loss_out, tuple):
            val_loss, loss_d = loss_out
        else:
            val_loss = loss_out
            loss_d = {'val_main_loss': loss_out}

        analysis(gt, sr)
        val_loss_dict = accum_loss_dict(val_loss_dict, loss_d)

    val_loss_dict = ave_ep_loss(val_loss_dict, i)
    if args.ddp:
        if args.log_metrics:
            _gathered_analysis: Union[List[AnalysisPanAcc], List[None]] = [
                None for _ in range(args.world_size)
            ]
            dist.gather_object(analysis, _gathered_analysis if is_main_process() else None)

        gathered_val_dict = [None for _ in range(args.world_size)]
        dist.gather_object(val_loss_dict, gathered_val_dict if is_main_process() else None)
        val_loss_dict = ave_multi_rank_dict(gathered_val_dict)

    acc_ave = analysis.acc_ave
    if is_main_process():
        # TODO: support different metrics
        if args.ddp and args.log_metrics:
            n = 0
            acc = {}  # {"SAM": 0.0, "ERGAS": 0.0, "PSNR": 0.0, "CC": 0.0, "SSIM": 0.0}
            for analysis in _gathered_analysis:
                for k, v in analysis.acc_ave.items():
                    acc[k] += v * analysis._call_n
                n += analysis._call_n
            for k, v in acc.items():
                acc[k] = v / n
            acc_ave = acc
        if logger is not None:
            # log validate curves
            if args.log_metrics:
                logger.log_curves(prefixed_dict_key(acc_ave, "val"), ep)
            logger.log_curve(val_loss / i, "val_loss", ep)
            for k, v in val_loss_dict.items():
                logger.log_curve(v, f'val_{k}', ep)

            # log validate image(last batch)
            if args.dataset not in ["hisi", 'far']:
                if gt.shape[0] > 8:
                    func = lambda x: x[:8, ...]
                    gt, lms, pan, sr = list(map(func, [gt, lms, pan, sr]))
                residual_image = res_image(gt, sr, exaggerate_ratio=100)  # [b, 1, h, w]
                _inc = sr.shape[1]  # wv3: 8, qb: 4, gf: 4
                logged_img = torch.cat(
                    [
                        lms,
                        pan.repeat(1, _inc, 1, 1),
                        sr,
                        residual_image.repeat(1, _inc, 1, 1),
                    ],
                    dim=0,
                )  # [3*b, c, h, w]
                logged_img = einops.rearrange(
                    logged_img, "(n k) c h w -> (k n) c h w", n=4
                )
                logger.log_images(logged_img, 4, "lms_pan_sr_res", ep)

            # print out eval information
            logger.print(
                ep_loss_dict2str(val_loss_dict),
                f"\n {dict_to_str(acc_ave)}" if args.log_metrics else ""
            )
    return (
        acc_ave,
        val_loss / i,
    )  # only rank 0 is reduced and other ranks are original data


def train(
        model: BaseModel,
        optim,
        criterion,
        warm_up_epochs,
        lr_scheduler,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int,
        eval_every_epochs: int,
        save_path: str,
        check_save_fn: Callable=None,
        logger: Union[WandbLogger, TensorboardLogger] = None,
        resume_epochs: int = 1,
        ddp=False,
        fp16=False,
        max_norm=None,
        grad_accum_ep=None,
        args=None,
):
    """
    train and val script
    :param network: Designed network, type: nn.Module
    :param optim: optimizer
    :param criterion: loss function, type: Callable
    :param warm_up_epochs: int
    :param lr_scheduler: scheduler
    :param train_dl: dataloader used in training
    :param val_dl: dataloader used in validate
    :param epochs: overall epochs
    :param eval_every_epochs: validate epochs
    :param save_path: model params and other params' saved path, type: str
    :param logger: Tensorboard logger or Wandb logger
    :param resume_epochs: when retraining from last break, you should pass the arg, type: int
    :param ddp: distribution training, type: bool
    :param fp16: float point 16, type: bool
    :param max_norm: max normalize value, used in clip gradient, type: float
    :param args: other args, see more in main.py
    :return:
    """
    print("start training!")
    network = model
    warm_up_scheduler = LinearWarmupScheduler(
        optim, 0, args.optimizer.lr, warm_up_epochs
    )
    world_size = args.world_size if ddp else None
    optim_val_loss = math.inf
    fp_scaler = amp.GradScaler() if fp16 else None
    for ep in range(resume_epochs, epochs + 1):
        if ddp:
            train_dl.sampler.set_epoch(ep)
            val_dl.sampler.set_epoch(ep)
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        # model training
        for i, (pan, ms, lms, gt) in enumerate(train_dl, 1):
            pan = pan.cuda().float()
            ms = ms.cuda().float()
            lms = lms.cuda().float()
            gt = gt.cuda().float()

            optim.zero_grad()
            with amp.autocast(enabled=fp16):
                sr, loss_out = network(ms, lms, pan, gt, criterion, mode="train")

                # if loss is hybrid, will return tensor loss and a dict
                if isinstance(loss_out, tuple):
                    loss, loss_d = loss_out
                else:
                    loss = loss_out
                ep_loss += loss
                ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)

            # update parameters
            step_loss_backward_partial = partial(
                step_loss_backward,
                optim=optim,
                network=network,
                max_norm=max_norm,
                loss=loss,
                fp16=fp16,
                fp_scaler=fp_scaler,
            )
            if grad_accum_ep is not None:
                grad_accm = ep % grad_accum_ep != 0
                context = model.no_sync if grad_accm and ddp else nullcontext
                with context():
                    step_loss_backward_partial(grad_accum=grad_accm)
                if grad_accm:
                    print("*" * 20, "grad_accm", "*" * 20)
            else:
                step_loss_backward_partial(grad_accum=False)

        # scheduler update
        if ep > warm_up_epochs:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(loss)
            else:
                lr_scheduler.step()
        else:
            warm_up_scheduler.step()

        # eval
        if ep % eval_every_epochs == 0:
            network.eval()
            val_acc_dict, val_loss = val(
                network, val_dl, criterion, logger, ep, optim_val_loss, args
            )
            network.train()
            params = {}
            try:
                params["model"] = network.module.state_dict()
            except Exception:  # any threw error
                params["model"] = network.state_dict()
            params["epochs"] = ep
            params["optim"] = optim.state_dict()
            params["lr_scheduler"] = lr_scheduler.state_dict()
            params["metrics"] = val_acc_dict
            if not args.save_every_eval:
                # TODO: find more straightforward way for managing the model saving
                # if (val_loss < optim_val_loss or check_save_fn(val_acc_dict)) and is_main_process():
                if check_save_fn(val_acc_dict) and is_main_process():
                    torch.save(params, save_path)
                    optim_val_loss = val_loss
                    logger.print(f"save params, best metric {check_save_fn.metric_name}: {check_save_fn.best_metric}")
            else:
                # TODO: 将同一个id的权重存在一个文件夹中，还需要考虑重新load训练的问题
                p = os.path.join(save_path, f'ep_{ep}.pth')
                torch.save(params, p)
                logger.print('save params')

            if ddp:
                dist.barrier()

        # print all info
        ep_loss /= i
        ep_loss_dict = ave_ep_loss(ep_loss_dict, i)
        lr = optim.param_groups[0]["lr"]
        if ddp:
            dist.reduce(ep_loss, 0)
            ep_loss_ranks_dict = [None for _ in range(world_size)]
            dist.gather_object(ep_loss_dict, ep_loss_ranks_dict if is_main_process() else None, 0)
            ep_loss_dict = ave_multi_rank_dict(ep_loss_ranks_dict)

        if logger is not None and ddp:
            if is_main_process():
                logger.log_curve(ep_loss / world_size, "train_loss", ep)
                for k, v in ep_loss_dict.items():
                    logger.log_curve(v / world_size, f'train_{k}', ep)
                logger.log_curve(lr, "lr", ep)
                logger.print(
                    f"[{ep}/{epochs}] lr: {lr} "
                    + ep_loss_dict2str(ep_loss_dict, world_size)
                )
        elif logger is None and ddp:
            if is_main_process():
                print(
                    f"[{ep}/{epochs}] lr: {lr} "
                    + ep_loss_dict2str(ep_loss_dict, world_size)
                )
        elif logger is not None and not ddp:
            logger.log_curve(ep_loss, "train_loss", ep)
            for k, v in ep_loss_dict.items():
                logger.log_curve(v, f'train_{k}', ep)
            logger.log_curve(lr, "lr", ep)
            logger.print(
                f"[{ep}/{epochs}] lr: {lr} "
                + ep_loss_dict2str(ep_loss_dict)
            )
        else:
            print(f"[{ep}/{epochs}] lr: {lr} "
                  + ep_loss_dict2str(ep_loss_dict))

        # watch network params(grad or data or both)
        if isinstance(logger, TensorboardLogger):
            logger.log_network(network, ep)
