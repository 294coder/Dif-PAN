from functools import partial
from typing import List
import torch
import torch.nn as nn


INNER_HOOK_OUTPUT = []


def get_inner_feature_hook(m: nn.Module, input, output, saved_list):
    if m.save_fm:
        saved_list.append(output)


def hook_model(model: nn.Module, hook, hook_module: nn.Module, saved_list):
    hook_handler = []
    for n, m in model.named_modules():
        if isinstance(m, hook_module):
            print(f"hook module {n}")
            h = m.register_forward_hook(
                partial(hook, saved_list=saved_list)
            )
            hook_handler.append(h)
    return h





    