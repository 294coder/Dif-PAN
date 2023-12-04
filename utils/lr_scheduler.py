import torch as th
import math


class WarmupCosineSchedule(object):
    """A simple warmup schedule with cosine decay."""

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.last_epoch = last_epoch
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return [
            base_lr
            * 0.5
            * (
                    1.0
                    + math.cos(
                math.pi
                * (self.last_epoch - self.warmup_steps)
                / (self.t_total - self.warmup_steps)
            )
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class WarmupLinearScheduler(object):
    """A simple warmup schedule with linear decay."""

    def __init__(
            self, optimizer, warmup_steps, t_total, last_epoch=-1, only_warmup=False
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.last_epoch = last_epoch
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        self.only_warmup = only_warmup

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        # descend lr linearly
        if not self.only_warmup:
            return [
                base_lr
                * (self.t_total - self.last_epoch)
                / (self.t_total - self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return list(map(lambda group: group["lr"], self.optimizer.param_groups))

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class MultiStepConstantScheduler(object):
    def __init__(self, optimizer, epoch_ms, lr_ms):
        assert len(epoch_ms) == len(lr_ms)
        self.optimizer = optimizer
        self.epoch_ms = epoch_ms
        self.lr_ms = lr_ms

        self.last_epoch = 0
        self.last_epoch_ms = -1

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        if self.last_epoch_ms < len(self.epoch_ms) - 1:
            if self.last_epoch >= self.epoch_ms[self.last_epoch_ms + 1]:
                for param_group in self.optimizer.param_groups:
                    self.last_epoch_ms += 1
                    # print(self.last_epoch, self.last_epoch_ms, self.lr_ms[self.last_epoch_ms])
                    param_group["lr"] = self.lr_ms[self.last_epoch_ms]


def get_lr_from_optimizer(optimizer):
    return optimizer.param_groups[0]["lr"]


class StepsAll:
    def __init__(self, *schedulers):
        self.schedulers = schedulers

    def step(self, *args, **kwargs):
        for s in self.schedulers:
            s.step(*args, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    net = th.nn.Linear(10, 10)
    optim = th.optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepConstantScheduler(
        optim, [10, 50, 100], [0.001, 0.0001, 0.00001]
    )
    lrs = []
    for i in range(200):
        scheduler.step()
        lr = get_lr_from_optimizer(optim)
        lrs.append(lr)
    plt.plot(range(200), lrs)
    fig = plt.gcf()
    fig.savefig("test.png")
