import os
from functools import partial
import math
from typing import Callable, List, Union

import accelerate.scheduler
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.utils.data import DataLoader
import accelerate
from tqdm import tqdm
from contextlib import nullcontext
from torch_ema import ExponentialMovingAverage
from transformers import get_scheduler

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
    model_params
)
from utils.log_utils import TensorboardLogger


def train(
        accelerator: accelerate.Accelerator,
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
    ema_net = ExponentialMovingAverage(parameters=[p for p in model.parameters() if p.requires_grad],
                                       decay=args.ema_decay)
    ema_net.to(next(model.parameters()).device)
    save_checker = lambda *check_args: check_save_fn(check_args[0]) if check_save_fn is not None else \
                    lambda val_acc_dict, val_loss, optim_val_loss: val_loss < optim_val_loss
    
    # if warm_up_epochs > 0:
    #     warm_up_scheduler = LinearWarmupScheduler(optim, 0, args.optimizer.lr, warm_up_epochs)
    
    

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.grad_accum_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Acclerator states
    # checkpointing_steps = args.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)
        
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * accelerator.grad_accum_steps

    world_size = args.world_size if ddp else None
    optim_val_loss = math.inf
    fp_scaler = amp.GradScaler() if fp16 else None
    
    logger.print(f">>> start training!")
    logger.print(f">>> Num examples = {len(train_dl)}")
    logger.print(f">>> Num Epochs = {args.num_train_epochs}")
    logger.print(f">>> Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.print(f">>> Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.print(f">>> Gradient Accumulation steps = {accelerator.grad_accum_steps}")
    logger.print(f">>> Total optimization steps = {args.max_train_steps}")
    
    for ep in range(resume_epochs, epochs + 1):
        if ddp:
            train_dl.sampler.set_epoch(ep)
            val_dl.sampler.set_epoch(ep)
        
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        # model training
        for i, (pan, ms, lms, gt) in enumerate(train_dl, 1):
            pan = pan.float()
            ms = ms.float()
            lms = lms.float()
            gt = gt.float()

            with accelerator.autocast() and accelerator.accumulate(model):
                sr, loss_out = model(ms, lms, pan, gt, criterion, mode="train")
                
                # if loss is hybrid, will return tensor loss and a dict
                if isinstance(loss_out, tuple):
                    loss, loss_d = loss_out
                else: loss = loss_out
                
                # if accelerator.sync_gradients:
                #     completed_steps += 1
                    
                ep_loss += loss
                if torch.isnan(loss).any():
                    raise ValueError(f">>> PROCESS {accelerator.device}: loss is nan")
                
                ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)

            # update parameters
            step_loss_backward_partial = partial(
                step_loss_backward,
                optim=optim,
                network=model,
                max_norm=max_norm,
                loss=loss,
                fp16=fp16,
                fp_scaler=fp_scaler,
                accelerator=accelerator,
            )
            
            optim.zero_grad()
            step_loss_backward_partial(grad_accum=False)
            
            ema_net.update()

        # scheduler update
        # FIXME: not support transformers ReduceLROnPlateau which is LRLambda, may be using inspect can fix?
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(loss)
        else:
            lr_scheduler.step(ep)
            
        # eval
        if ep % eval_every_epochs == 0:
            model.eval()
            with ema_net.average_parameters():
                val_acc_dict, val_loss = val(model, val_dl, criterion, logger, ep, optim_val_loss, args)
            model.train()
            
            def collect_params():
                params = {}
                # params["model"] = model_params(model)
                params["ema_model"] = ema_net.state_dict()  # TODO: contain on-the-fly params, find way to remove and not affect the load
                params["epochs"] = ep
                params["metrics"] = val_acc_dict
                
                return params
            
            params = collect_params()
            if save_checker(val_acc_dict, val_loss, optim_val_loss) and accelerator.is_main_process:
                # torch.save(params, save_path)
                accelerator.save(params, save_path, safe_serialization=True)
                optim_val_loss = val_loss
                logger.print("save params")
                
        if isinstance(args.checkpoint_every_n, int):
            if ep % args.checkpoint_every_n == 0:
                output_dir = f"ep_{ep}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
            
        accelerator.wait_for_everyone()
            
        # print all info
        ep_loss /= i
        ep_loss_dict = ave_ep_loss(ep_loss_dict, i)
        lr = optim.param_groups[0]["lr"]
        if accelerator.use_distributed:
            accelerator.reduce(ep_loss)
            accelerator.gather_for_metrics(ep_loss_dict)
            ep_loss_dict = ave_multi_rank_dict(ep_loss_dict)

        if logger is not None and accelerator.use_distributed:
            if accelerate.is_main_process(): 
                logger.log_curve(ep_loss / world_size, "train_loss", ep)
                for k, v in ep_loss_dict.items():
                    logger.log_curve(v / world_size, f'train_{k}', ep)
                logger.log_curve(lr, "lr", ep)
                logger.print(
                    f"[{ep}/{epochs}] lr: {lr} "
                    + ep_loss_dict2str(ep_loss_dict, world_size)
                )
        elif logger is None and accelerator.use_distributed:
            if accelerate.is_main_process():
                print(
                    f"[{ep}/{epochs}] lr: {lr} "
                    + ep_loss_dict2str(ep_loss_dict, world_size)
                )
        elif logger is not None and not accelerator.use_distributed:
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
            logger.log_network(model, ep)
