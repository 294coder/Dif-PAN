from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict

class HybridL1SSIM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1)):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss


class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class LossWarpper(torch.nn.Module):
    def __init__(self, weighted_ratio=(1.0, 1.0), **losses):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(losses.keys())
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt):
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            loss += l
            d_loss[n] = l
        return loss, d_loss


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()


def get_loss(loss_type):
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "l1ssim":
        criterion = HybridL1SSIM(channel=8, weighted_r=(1.0, 0.1))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(channel=31, weighted_r=(1.0, 1.0))
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


if __name__ == "__main__":
    loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    x = torch.randn(1, 31, 64, 64, requires_grad=True)
    y = x + torch.randn(1, 31, 64, 64) / 10
    l = loss(x, y)
    l.backward()
    print(l)
    print(x.grad)

