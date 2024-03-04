import math
from typing import Union
from copy import deepcopy

import torch
import torch.nn as nn


# from model.base_model import BaseModel


def hook_model(model: nn.Module, saved_tensor, hook_class):
    def feature_hook(_, input, output):
        # forward hook
        saved_tensor.append([input, output])

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, hook_class):
            hooks.append(m.register_forward_hook(feature_hook))
    return model, hooks


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, loss, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if "conv" in k and "weight" in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = loss + sum(regularizations)
        return loss


def variance_scaling_initializer(tensor):
    # stole it from woo-xiao.
    # thanks
    def calculate_fan(shape, factor=2.0, mode="FAN_IN", uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == "FAN_IN":
            # Count only number of input connections.
            n = fan_in
        elif mode == "FAN_OUT":
            # Count only number of output connections.
            n = fan_out
        elif mode == "FAN_AVG":
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev
    
def model_params(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    elif isinstance(model, torch._dynamo.eval_frame.OptimizedModule):  # torch.compile model
        model = model._orig_mod
    return model.state_dict()

def model_device(model: Union[nn.Module, nn.DataParallel, 
                              nn.parallel.DistributedDataParallel,
                              torch._dynamo.eval_frame.OptimizedModule]):
    params = model.parameters()
    p0 = next(params)
    return p0.device


def clip_norm(max_norm, network, fp_scaler=None, optim=None):
    if fp_scaler is not None:
        fp_scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)


def step_loss_backward(
        optim,
        network=None,
        max_norm=None,
        loss=None,
        fp16=False,
        fp_scaler=None,
        grad_accum=False,
):
    """

    :param optim: optimizer. type: optim.Optimizer
    :param network: instanced network. type: nn.Module
    :param max_norm: clip norm. type: float
    :param loss: float
    :param fp16: bool
    :param fp_scaler: mix-precision scaler
    :return:
    """
    if (fp16 and fp_scaler is None) or (not fp16 and fp_scaler is not None):
        raise ValueError("fp16 and grad_scaler should be set together")
    if max_norm is not None and network is None:
        raise ValueError("max_norm is set, network should be set")

    if fp16:
        fp_scaler.scale(loss).backward()
        if max_norm is not None:
            clip_norm(max_norm, network, fp_scaler, optim)
        if not grad_accum:
            fp_scaler.step(optim)
            fp_scaler.update()
    else:
        loss.backward()
        if max_norm is not None:
            clip_norm(max_norm, network)
        if not grad_accum:
            optim.step()


class EMAModel(object):
    def __init__(self, model, ema_ratio=0.9999):
        super().__init__()
        self.model = model
        self.ema_ratio = ema_ratio
        self.ema_model = deepcopy(model)

    def update(self):
        for ema_p, now_p in zip(self.ema_model.state_dict(), self.model.state_dict()):
            ema_p.data = ema_p.data * self.ema_ratio + now_p.data * (1 - self.ema_ratio)

    def ema_model_state_dict(self):
        try:
            return self.ema_model.module.state_dict()
        except:
            return self.ema_model.state_dict()

# def ema_model(prev_model: Union[nn.Module, BaseModel], model: Union[nn.Module, BaseModel], ema_rate=0.999):
#     prev_params = prev_model.state_dict()
#     now_params = model.state_dict()
#     for prev_p, now_p in zip(prev_params, now_params):
#         now_params.data = prev_p.data * ema_rate + now_p.data * (1 - ema_rate)
