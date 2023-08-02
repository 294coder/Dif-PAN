# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2023/4/10 12:37
# @Author  : Xiao Wu
# @reference:
#
import itertools

import numpy as np
from utils.misc import dict_to_str


class AnalysisFLIRAcc(object):
    def __init__(self, unorm=True):
        self.metric_fn = analysis_Reference_fast
        self.unorm_factor = 255 if unorm else 1

        # acc tracker
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = {}
        self.sumed_acc = {}

    def _average_acc(self, d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave

    def drop_dim(self, x):
        assert x.ndim == 3 and x.shape[
            0] == 1, f'x.dim should be 4, but got {x.ndim} or x.shape[1] should be 1 but got {x.shape[0]}'
        # more safe. but unsafe setting is x.squeeze()
        return x[0]

    def dict_items_sum(self, b_acc_d: list):
        sum_d = {}
        for acc_d in b_acc_d:
            for k, v in acc_d.items():
                sum_d[k] = sum_d.get(k, 0) + v
        return sum_d

    def average_all(self, sum_d, b):
        self._call_n += b
        for k, v in sum_d.items():
            self.sumed_acc[k] = self.sumed_acc.get(k, 0) + v
            self.acc_ave[k] = self.sumed_acc[k] / self._call_n

    def one_batch_call(self, gt, pred):
        assert gt.shape[-2:]==pred.shape[-2:], f'gt and pred should have same shape,' \
            f'but got gt.shape[-2:]=={gt.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
        
        b = gt.shape[0]
        gt = gt * self.unorm_factor
        pred = pred * self.unorm_factor

        # input shapes are [B, C, H, W]
        # gt is [b, 2, h, w]
        # pred is [b, 1, h, w]
        batch_acc_d = []
        for i, (vi, ir, f) in enumerate(zip(*gt.chunk(2, 1), pred), 1):
            vi, ir, f = map(self.drop_dim, (vi, ir, f))
            acc_d = self.metric_fn(f, ir, vi)
            batch_acc_d.append(acc_d)

        sum_d = self.dict_items_sum(batch_acc_d)
        self._acc_d = sum_d
        self.average_all(sum_d, b)

    def __call__(self, gt, pred):
        self.one_batch_call(gt, pred)

    def print_str(self):
        return dict_to_str(self.acc_ave)

    @property
    def last_acc(self):
        return self._acc_d


#########
# metric helpers
#########

import torch
from torch.nn import functional as F


def cal_PSNR(A, B, F):
    [m, n] = F.shape
    MSE_AF = torch.sum((F - A) ** 2) / (m * n)
    MSE_BF = torch.sum((F - B) ** 2) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * torch.log10(255 / torch.sqrt(MSE))

    return PSNR


def cal_SD(F):
    [m, n] = F.shape
    u = torch.mean(F)

    # 原版 is wrong
    # tmp = (F - u).numpy()
    # tmp2 = np.round(tmp.clip(0)).astype(np.uint16)
    # SD = np.sqrt(np.sum((tmp2 ** 2).clip(0, 255)) / (m * n))

    SD = torch.sqrt(torch.sum((F - u) ** 2) / (m * n))

    return SD


def cal_EN(I):
    p = torch.histc(I, 256)
    p = p[p != 0]
    p = p / torch.numel(I)
    E = -torch.sum(p * torch.log2(p))
    return E


def cal_SF(MF):
    # RF = MF[]#diff(MF, 1, 1);
    [m, n] = MF.shape
    RF = MF[:m - 1, :] - MF[1:]
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF = MF[:, :n - 1] - MF[:, 1:]  # diff(MF, 1, 2)
    CF1 = torch.sqrt(torch.mean(CF ** 2))
    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)

    return SF


def analysis_Reference_fast(image_f, image_ir, image_vis):
    # shapes are [h, w], channel is 1
    # image_f: 0-255
    # image_ir: 0-255
    # image_vis: 0-255

    psnr = cal_PSNR(image_ir, image_vis, image_f)
    SD = cal_SD(image_f)
    EN = cal_EN(image_f)
    SF = cal_SF(image_f / 255.0)
    AG = cal_AG(image_f)
    SSIM = cal_SSIM(image_ir, image_vis, image_f)
    # print(psnr, SD, EN, SF, AG, SSIM)

    return dict(
        PSNR=psnr.item(),
        EN=EN.item(),
        SD=SD.item(),
        SF=SF.item(),
        AG=AG.item(),
        SSIM=SSIM.item()
    )


def cal_AG(img):
    if len(img.shape) == 2:
        [r, c] = img.shape
        [dzdx, dzdy] = torch.gradient(img)
        s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        g = torch.sum(s) / ((r - 1) * (c - 1))

    else:
        [r, c, b] = img.shape
        g = torch.zeros(b)
        for k in range(b):
            band = img[:, :, k]
            [dzdx, dzdy] = torch.gradient(band)
            s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
            g[k] = torch.sum(s) / ((r - 1) * (c - 1))
    return torch.mean(g)


def _ssim(img1, img2):
    device = img1.device
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11

    # 不加这个,对应matlab的quality_assess的ssim指标
    # pad_size = [window_size//2]*4
    # img1 = F.pad(img1, mode='replicate', pad=pad_size)
    # img2 = F.pad(img2, mode='replicate', pad=pad_size)

    window = create_window(window_size, sigma, channel).to(device)
    mu1 = F.conv2d(img1, window, groups=channel)  # , padding=window_size // 2
    mu2 = F.conv2d(img2, window, groups=channel)  # , padding=window_size // 2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq  # , padding=window_size // 2
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq  # , padding=window_size // 2
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2  # , padding=window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)


import math
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def cal_SSIM(im1, im2, image_f):
    # h, w -> 2, h, w -> b, 2, h, w
    img_Seq = torch.stack([im1, im2])
    image_f = image_f.unsqueeze(0).repeat([img_Seq.shape[0], 1, 1])
    return torch.mean(_ssim(img_Seq.unsqueeze(0) / 255.0, image_f / 255.0))


if __name__ == '__main__':
    # from scipy import io as sio
    #
    # data = sio.loadmat('./test_IRF.mat')
    # im1 = data['image1']
    # im2 = data['image2']
    # image_f = data['image_f']
    # print(im1.shape, im2.shape, image_f.shape, im1.max(), im2.max(), image_f.max())
    # im1 = torch.from_numpy(im1).float() * 255
    # im2 = torch.from_numpy(im2).float() * 255
    # image_f = torch.from_numpy(image_f).float()
    #
    # analysis_Reference_fast(image_f, im1, im2)

    # f = torch.randint(0, 255, (256, 256), dtype=torch.float)
    # vi = torch.randint(0, 255, (256, 256), dtype=torch.float)
    # ir_RS = torch.randint(0, 255, (256, 256), dtype=torch.float)

    # analysis_Reference_fast(f, vi, ir_RS)

    gt = (torch.randn(2, 2, 256, 256) + 1) / 2
    sr = (torch.randn(2, 1, 256, 256) + 1) / 2
    analyser = AnalysisFLIRAcc()

    analyser(gt, sr)
    print(analyser.print_str())

    analyser(gt, sr)
    print(analyser.print_str())
