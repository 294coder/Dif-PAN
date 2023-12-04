from collections import OrderedDict
import os
from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torchvision as tv


def exist(b):
    if b is not None:
        return True
    else:
        return False


def default(a, b):
    if exist(b):
        return b
    else:
        return a


def grad_clip(params, mode: str = "value", value: float = None, **kwargs) -> None:
    """do a gradient clipping

    Args:
        params (tensor): model params
        mode (str, optional): 'value' or 'norm'. Defaults to 'value'.
    """
    assert mode in ["value", "norm"], "mode should be @value or @norm"
    if mode == "norm":
        nn.utils.clip_grad.clip_grad_norm_(parameters=params, max_norm=value, **kwargs)
    else:  # mode == 'value'
        nn.utils.clip_grad.clip_grad_value_(parameters=params, clip_value=value)


def model_froze(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)


def dict_to_str(d):
    n = len(d)
    def func(
        k, v): return f"{k}: {v.item() if isinstance(v, torch.Tensor) else v}"
    s = ""
    for i, (k, v) in enumerate(d.items()):
        s += func(k, v) + (", " if i < n - 1 else "")
    return s


def to_numpy(*args):
    l = []
    for i in args:
        l.append(i.detach().cpu().numpy())
    return l


def norm_data_range(x):
    """norm input to [-1, 1]

    Args:
        x (torch.Tensor): input

    Returns:
        torch.Tensor: output with data ranging in [-1, 1]
    """
    x = x - x.min()
    x = x / x.max()
    x = 2 * x - 1
    return x


def unnorm_data_range(x):
    """unnormalized input to data range [0, 1]

    Args:
        x (Tensor): input data ranging in [-1, 1]

    Returns:
        Tensor: output data ranging in [0, 1]
    """
    return (x + 1) / 2


def model_load(
    path: str, model: nn.Module, strict: bool = True, device: str = None
) -> nn.Module:
    """model load parameters

    Args:
        path (str): checkpoint path
        model (nn.Module): model instance
        strict (bool, optional): strictly load. Defaults to True.

    Returns:
        nn.Module: _description_
    """
    if not exist(device):
        device = next(model.parameters()).device
    params = torch.load(path, map_location=device)
    try:
        model.load_state_dict(params, strict=strict)
    except Exception:
        try:
            odict = OrderedDict()
            # remove module.
            for k, v in params.items():
                k.replace("module.", "")
                odict[k] = v

            model.load_state_dict(odict, strict=strict)
        except Exception:
            if not strict:
                model = _regardless_keys_unmatch_shape_unmatch(model, params)
            else:
                raise RuntimeError("strict is True, but model load failed")

    return model


def _regardless_keys_unmatch_shape_unmatch(model, state_dict):
    state_dict1 = model.state_dict()
    state_dict2 = state_dict
    for k, v in state_dict1.items():
        if k in state_dict2.keys():
            if v.shape == state_dict2[k].shape:
                state_dict1[k] = state_dict2[k]
    model.load_state_dict(state_dict1)
    return


def list_tensor2_list_list(list_tensor, n, mode="time"):
    """a list of tensor like [ta, tb, tc, td, tf, te] ->
    [[ta, tb, tc], [td, tf, te]] when n is 3 and mode is 'time';
    [[ta, td], [tb, td], [tc, te]] when n is 3 and mode is 'size'

    Args:
        list_tensor (list): a list of tensor
        n (int): an int
        mode (str): time or size
    """

    num = len(list_tensor)
    assert num % n == 0

    out = []
    if mode == "time":
        for i in range(num // n):
            sub_list = list_tensor[i * n: (i + 1) * n]
            out.append(sub_list)
    elif mode == "size":
        for i in range(n):
            sub_list = list_tensor[i::n]
            out.append(sub_list)
    else:
        raise NotImplementedError(f"mode {mode} is not supported")
    return out


def concat_dif_timesteps_tensor(
    tensor_list, n: int, out_size
):
    cat_tensor = list_tensor2_list_list(tensor_list, n, mode="size")
    ts = []
    for i in range(n):
        t = cat_tensor[i]
        t = torch.cat(t, dim=1)
        t = nn.functional.interpolate(
            t, out_size, mode="bilinear", align_corners=True)
        ts.append(t)
    return torch.cat(ts, dim=1)


def img_batch2one_img(batched_img: torch.Tensor) -> torch.Tensor:
    # batch_img shape: (batch, channel, height, width)
    # return img on cpu, shape: (height, width, channel)

    b = batched_img.shape[0]
    if batched_img.shape[1] > 3:
        s = [0, 2, 4]
    else:
        s = slice(None)
    img = tv.utils.make_grid(
        batched_img.detach().cpu(), nrow=int(b ** 0.5), padding=0, normalize=True
    ).permute(1, 2, 0)[..., s]
    return img


def path_legal_checker(path, is_file=True):
    if is_file:
        path2 = os.path.dirname(path)
    else:
        path2 = path
    if not os.path.exists(path2):
        os.makedirs(path2)
        print("path not exist, create it: ", path2)
    return path


def compute_iters(size, bs, drop_last=False):
    fp_iters = size / bs
    int_iter = np.ceil(fp_iters).astype('int')
    last = int((fp_iters - int_iter) > 0.) if not drop_last else 0
    return int_iter + last


if __name__ == "__main__":
    # _a = torch.randn(1, 8, 64, 64)
    # _b = torch.randn(1, 8, 32, 32)
    # _c = torch.randn(1, 8, 16, 16)
    # a = [_a, _b, _c, _a, _b, _c]
    # print(concat_dif_timesteps_tensor(a, n=3, out_size=64).shape)
    pass
