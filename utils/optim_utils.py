from typing import Union
import numpy as np
import torch.nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


class IdentityScheduler:
    # a placeholder for lr_scheduler or weight_decay_scheduler
    def __init__(self, optim, **kwargs):
        self.optim = optim
        self.kwargs = kwargs

    def step(self, *args, **kwargs):
        pass

    def state_dict(self):
        return self.kwargs





def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    copy from DINO. manually set learning lr every iteration.
    note that there is only half epoch of cosine, which means learning rate will not
    go back to the original.
    :param base_value:
    :param final_value:
    :param epochs:
    :param niter_per_ep:
    :param warmup_epochs:
    :param start_warmup_value:
    :return:
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class LinearWarmupScheduler:
    def __init__(self, opt: optim.Optimizer, init_value, warmup_value, warmup_epochs):
        self.opt = opt
        self.init_value = init_value
        self.warmup_value = warmup_value
        self.warmup_epochs = warmup_epochs
        self.values = np.linspace(init_value, warmup_value, warmup_epochs)
        self.now_index = 0

    def step(self):
        self.opt.param_groups[0]["lr"] = self.values[self.now_index]
        self.now_index += 1


def get_scheduler(optim, **kwargs):
    """
    get lr_scheduler or weight_decay_scheduler
    Args:
        optim: optimizer
        **kwargs: a dict containing type of scheduler and its arguments

    Returns: a scheduler

    """
    name = kwargs["name"]
    kwargs.pop("name")
    if name == "cos_anneal":
        return CosineAnnealingLR(optim, **kwargs)
    elif name == "cos_anneal_restart":
        return CosineAnnealingWarmRestarts(optim, **kwargs)
    elif name == "multi_step":
        return MultiStepLR(optim, **kwargs)
    elif name == "plateau":
        return ReduceLROnPlateau(optim, **kwargs)
    elif name == "identity":
        return IdentityScheduler(optim, **kwargs)
    else:
        raise NotImplementedError


def get_optimizer(params, **kwargs):
    name = kwargs["name"]
    kwargs.pop("name")
    if name == "sgd":
        return optim.SGD(params, **kwargs)
    elif name == "adam":
        return optim.Adam(params, **kwargs)
    elif name == "adamw":
        return optim.AdamW(params, **kwargs)
    elif name == "lion":
        from lion_pytorch import Lion
        return Lion(params, **kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torch.nn as nn

    init_lr = 1e-3
    final_lr = 1e-6
    epochs = 500
    nither_per_ep = int(np.ceil(3000 // 16))  # len(datasets) / batch_size
    warm_epochs = 80
    start_warmup_value = init_lr
    cos_sche = cosine_scheduler(
        init_lr, final_lr, epochs, nither_per_ep, warm_epochs, start_warmup_value
    )
    plt.plot(list(map(lambda x: x / nither_per_ep, range(len(cos_sche)))), cos_sche)
    # plt.show()

    # torch cosine annealing lr scheduler
    net = nn.Sequential(nn.Linear(8, 64))
    optimizer = optim.AdamW(net.parameters(), lr=init_lr)
    cos_sche2 = CosineAnnealingLR(optimizer, epochs - warm_epochs, final_lr)
    lr = []
    for i in range(500):
        l = optimizer.param_groups[0]["lr"]
        lr.append(l)
        if i > warm_epochs:
            cos_sche2.step()
    plt.plot(range(500), lr)
    plt.show()
