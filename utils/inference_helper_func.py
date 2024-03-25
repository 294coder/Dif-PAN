# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2021/10/15 17:53
# @Author  : Xiao Wu
# reference:
import inspect
from typing import Tuple, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch import Tensor, nn
from tqdm import tqdm

from .visualize import viz_batch, res_image
from .metric import AnalysisPanAcc
from model.base_model import BaseModel, PatchMergeModule
from utils import get_local

def has_patch_merge_model(model: nn.Module):
    return (hasattr(model, '_patch_merge_model')) or (hasattr(model, 'patch_merge_model'))


def patch_merge_in_val_step(model):
    return 'patch_merge' in list(inspect.signature(model.val_step).parameters.keys())


@torch.no_grad()
@torch.inference_mode()
def unref_for_loop(model,
                   dl: DataLoader,
                   device,
                   split_patch=False,
                   **patch_merge_module_kwargs):
    all_sr = []
    try:
        spa_size = tuple(dl.dataset.lms.shape[-2:])
    except AttributeError:
        spa_size = tuple(dl.dataset.rgb.shape[-2:])
    
    inference_bar = tqdm(enumerate(dl, 1), dynamic_ncols=True, total=len(dl))
    analysis = AnalysisPanAcc(ratio=patch_merge_module_kwargs.get('ergas_ratio', 4), ref=False,
                              sensor=patch_merge_module_kwargs.get('sensor', 'DEFAULT'),
                              default_max_value=patch_merge_module_kwargs.get('default_max_value', None))
    
    if split_patch:
        # check if has the patch merge model
        if not (has_patch_merge_model(model) or patch_merge_in_val_step(model)):
            # assert bs == 1, 'batch size should be 1'
            
            # warp the model into PatchMergeModule
            model = PatchMergeModule(net=model, device=device, **patch_merge_module_kwargs)
            
    for i, (pan, ms, lms) in inference_bar:
        pan, ms, lms = pan.to(device).float(), ms.to(device).float(), lms.to(device).float()
        # split the image into several patches to avoid gpu OOM
        if split_patch:
            input = (ms, lms, pan)
            if hasattr(model, 'forward_chop'):
                # split the image into several patches to avoid gpu OOM
                # pan_nc = pan.size(1)
                # ms_nc = ms.size(1)
                # input = (
                #     F.interpolate(ms, size=lms.shape[-1], mode='bilinear', align_corners=True),
                #     lms,
                #     torch.cat([pan, torch.zeros(bs, ms_nc - pan_nc, *spa_size).to(device)], dim=1)
                # )
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model):
                sr = model.val_step(*input, patch_merge=True)
            else:
                raise NotImplemented('model should have @forward_chop or patch_merge arg in @val_step')
        else:
            if patch_merge_in_val_step(model):
                sr = model.val_step(ms, lms, pan, False)
            else:
                sr = model.val_step(ms, lms, pan)
        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)
        
        analysis(sr, ms, lms, pan)
        
        viz_batch(sr.detach().cpu(), suffix='sr', start_index=i)
        viz_batch(ms.detach().cpu(), suffix='ms', start_index=i)
        viz_batch(pan.detach().cpu(), suffix='pan', start_index=i)
        
    print(analysis.print_str())

    return all_sr


@torch.no_grad()
@torch.inference_mode()
def ref_for_loop(model,
                 dl,
                 device,
                 split_patch=False,
                 ergas_ratio=4,
                 residual_exaggerate_ratio=100,
                 **patch_merge_module_kwargs):
    analysis = AnalysisPanAcc(ergas_ratio)
    all_sr = []
    inference_bar = tqdm(enumerate(dl, 1), dynamic_ncols=True, total=len(dl))

    if not (has_patch_merge_model(model) or patch_merge_in_val_step(model)):
            # assert bs == 1, 'batch size should be 1'
            
            # warp the model into PatchMergeModule
            model = PatchMergeModule(net=model, device=device, **patch_merge_module_kwargs)
    for i, (pan, ms, lms, gt) in inference_bar:
        pan, ms, lms, gt = pan.to(device).float(), ms.to(device).float(), lms.to(device).float(), gt.to(device).float()
        # split the image into several patches to avoid gpu OOM
        if split_patch:
            input = (ms, lms, pan)
            if hasattr(model, 'forward_chop'):
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model):
                sr = model.val_step(*input, patch_merge=True)
            else:
                raise NotImplemented('model should have @forward_chop or patch_merge arg in @val_step')
        else:
            if patch_merge_in_val_step(model):
                sr = model.val_step(ms, lms, pan, False)
            else:
                sr = model.val_step(ms, lms, pan)
                
        cache = get_local().cache
        attns = cache['MSReversibleRefine.forward']
        # attns = cache['FirstAttn.forward']
        
        torch.save(attns, f'/volsparse1/czh/exps/fcformer-bk/visualized_img/attns/attns_{i}.pth')
        print('saved pth file...')
        get_local.clear()
                
        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)

        analysis(gt, sr)

        res = res_image(gt, sr, exaggerate_ratio=residual_exaggerate_ratio)
        viz_batch(sr.detach().cpu(), suffix='sr', start_index=i)
        viz_batch(gt.detach().cpu(), suffix='gt', start_index=i)
        viz_batch(ms.detach().cpu(), suffix='ms', start_index=i)
        viz_batch(pan.detach().cpu(), suffix='pan', start_index=i)
        viz_batch(res.detach().cpu(), suffix='residual', start_index=i)

        # print(f'PSNR: {psnr}, SSIM: {ssim}')

    print(analysis.print_str())

    return all_sr


"""
NOTE:
当图片过大，无法放进gpu甚至内存中运行时，使用此helper function:
    继承PatchMergeModule，val的时候将val_step里的self()或者self.forward()换成self.fordward_chop
    输入参数我是直接传的args配置，你可以自己改一下，args.patch_size要求是个tuple,然后必须满足卷积的公式，卷积公式限制了一些patch大小
    self.forward_chop
    args.patch_size=(H,W)，是允许长方形的，不过没必要用到，是之前做双目超分的时候这样效果比较好
"""


# implement tf.gather_nd() in pytorch
def gather_nd(tensor, indexes, ndim):
    '''
    inputs = torch.randn(1, 3, 5)
    base = torch.arange(3)
    X_row = base.reshape(-1, 1).repeat(1, 5)
    lookup_sorted, indexes = torch.sort(inputs, dim=2, descending=True)
    print(inputs)
    print(indexes, indexes.shape)
    # print(gathered)
    print(gather_nd(inputs, indexes, [1, 2]))
    '''
    if len(ndim) == 2:
        base = torch.arange(indexes.shape[ndim[0]])
        row_index = base.reshape(-1, 1).repeat(1, indexes.shape[ndim[1]])
        gathered = tensor[..., row_index, indexes]
    elif len(ndim) == 1:
        base = torch.arange(indexes.shape[ndim[0]])
        gathered = tensor[..., base, indexes]
    else:
        raise NotImplementedError
    return gathered


# class PatchMergeModule(nn.Module):
#     def __init__(self,
#                  crop_batch_size=1,
#                  patch_size=128,
#                  scale=1,
#                  hisi=False,
#                  device='cuda:0',
#                  bs_axis_merge=True):
#         super().__init__()
#         if bs_axis_merge:
#             self.split_func = torch.split
#         else:
#             self.split_func = lambda x, _, dim: [x]
#         self.crop_batch_size = crop_batch_size
#         self.patch_size = patch_size
#         self.scale = scale
#
#     def forward(self, x):
#
#         return x, x
#
#     def forward_chop(self, *x, shave=12, **kwargs):
#         # 不存在输入张量不一样的情况, 如不一样请先处理成一样的再输入
#         # 但输出存在维度不一样的情况, 因为网络有多层并且每层尺度不同, 进行分开处理: final_output, intermediate_output
#         # TODO: bug: 1. 输出的张量尺寸不一样的时候 无法进行fold操作, 一是因为参数, 二是可能还原不回去
#         #            2. unfold的在重叠切的时候是共享原始数据的，所以后者会覆盖前者
#
#         x = torch.cat(x, dim=0)
#         split_func = self.split_func
#         # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         x.cpu()
#         batchsize = self.crop_batch_size
#         patch_size = self.patch_size
#
#         b, c, h, w = x.size()
#         if isinstance(patch_size, (list, tuple)):
#             (h_padsize, w_padsize) = patch_size
#             hshave, wshave = h_padsize // 2, w_padsize // 2
#         else:
#             h_padsize = w_padsize = int(patch_size)
#             hshave = wshave = patch_size // 2
#         # print(self.scale, self.idx_scale)
#         scale = self.scale  # self.scale[self.idx_scale]
#
#         h_cut = (h - h_padsize) % (int(hshave / 2))
#         w_cut = (w - w_padsize) % (int(wshave / 2))
#
#         x_unfold = F.unfold(x, (h_padsize, w_padsize), stride=(int(hshave / 2), wshave // 2)).permute(2, 0,
#                                                                                                       1).contiguous()
#
#         ################################################
#         # 最后一块patch单独计算
#         ################################################
#
#         x_hw_cut = x[..., (h - h_padsize):, (w - w_padsize):]
#         y_hw_cut = self.forward(*[s.cuda() for s in split_func(x_hw_cut, [1, 1], dim=0)], **kwargs)
#         y_hw_cut = self.forward(*[s.cuda() for s in split_func(x_hw_cut, [1, 1], dim=0)], **kwargs)
#
#         x_h_cut = x[..., (h - h_padsize):, :]
#         x_w_cut = x[..., :, (w - w_padsize):]
#         y_h_cut = self.cut_h(x_h_cut, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
#                              **kwargs)
#         y_w_cut = self.cut_w(x_w_cut, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
#                              **kwargs)
#         # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # 左上patch单独计算，不是平均而是覆盖
#         ################################################
#
#         x_h_top = x[..., :h_padsize, :]
#         x_w_top = x[..., :, :w_padsize]
#         y_h_top = self.cut_h(x_h_top, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
#                              **kwargs)
#         y_w_top = self.cut_w(x_w_top, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
#                              **kwargs)
#
#         # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # img->patch，最大计算crop_s个patch，防止bs*p*p太大
#         ################################################
#
#         x_unfold = x_unfold.view(x_unfold.size(0), -1, c, h_padsize, w_padsize)
#         y_unfold = []
#
#         x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
#         x_unfold.cuda()
#         for i in range(x_range):
#             res = self.forward(
#                 *[s[:, 0, ...] for s in split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)],
#                 **kwargs)
#             y_unfold.append(res)
#             # y_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)]
#             #     , **kwargs)])
#             # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#
#         # for i, s in enumerate(zip(*y_unfold)):
#         #     if i < len(y_unfold):
#         #         y_unfold[i] = s
#         #     else:
#         #         y_unfold.append(s)
#         if isinstance(y_unfold[0], tuple):
#             y_unfold_out = [None] * len(y_unfold[0])
#             for i, s in enumerate(zip(*y_unfold)):
#                 y_unfold_out[i] = s
#             y_unfold = y_unfold_out
#             y_hw_cut = [s.cpu() for s in y_hw_cut]
#         else:
#             y_unfold = [y_unfold]
#             y_hw_cut = [y_hw_cut.cpu()]
#
#         out = []
#         for s_unfold, s_h_top, s_w_top, s_h_cut, s_w_cut, s_hw_cut in zip(y_unfold, y_h_top, y_w_top, y_h_cut, y_w_cut,
#                                                                           y_hw_cut):
#             s_unfold = torch.cat(s_unfold, dim=0).cpu()
#
#             y = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                        ((h - h_cut) * scale, (w - w_cut) * scale), (h_padsize * scale, w_padsize * scale),
#                        stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))
#             # 312， 480
#             # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#             ################################################
#             # 第一块patch->y
#             ################################################
#             y[..., :h_padsize * scale, :] = s_h_top
#             y[..., :, :w_padsize * scale] = s_w_top
#             # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#             s_unfold = s_unfold[...,
#                        int(hshave / 2 * scale):h_padsize * scale - int(hshave / 2 * scale),
#                        int(wshave / 2 * scale):w_padsize * scale - int(wshave / 2 * scale)].contiguous()
#
#             s_inter = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                              ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
#                              (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
#                              stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))
#             # 1，3，750，540
#             #
#             s_ones = torch.ones(s_inter.shape, dtype=s_inter.dtype)
#             divisor = F.fold(F.unfold(s_ones, (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
#                                       stride=(int(hshave / 2 * scale), int(wshave / 2 * scale))),
#                              ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
#                              (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
#                              stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))
#
#             s_inter = s_inter.cpu() / divisor
#             # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#             ################################################
#             # 第一个半patch
#             ################################################
#             y[..., int(hshave / 2 * scale):(h - h_cut) * scale - int(hshave / 2 * scale),
#             int(wshave / 2 * scale):(w - w_cut) * scale - int(wshave / 2 * scale)] = s_inter
#             # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#             y = torch.cat([y[..., :y.size(2) - int((h_padsize - h_cut) / 2 * scale), :],
#                            s_h_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
#             # 图分为前半和后半
#             # x->y_w_cut
#             # model->y_hw_cut TODO:check
#             y_w_cat = torch.cat([s_w_cut[..., :s_w_cut.size(2) - int((h_padsize - h_cut) / 2 * scale), :],
#                                  s_hw_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
#             y = torch.cat([y[..., :, :y.size(3) - int((w_padsize - w_cut) / 2 * scale)],
#                            y_w_cat[..., :, int((w_padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
#             out.append(y.cuda())
#             # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#             # plt.show()
#
#         return out  # y.cuda()
#
#     def cut_h(self, x_h_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):
#         split_func = self.split_func
#         # 1, 3, 30, 600 -> (30, 120), s=(7, 30)
#         # 1,3*30*120, 17 -> 17, 1, 3, 30, 120
#         # N = [(H - k_h + 2*pad_h) / s_h + 1] * [(W - k_w + 2*pad_w) / s_w + 1]
#         #   = [1+(30 - 30) / 7] * [1+(600 - 120) / 30] = 1*17 = 17
#         x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=(int(shave[0] / 2), int(shave[1] / 2))).permute(2, 0,
#                                                                                                            1).contiguous()  # transpose(0, 2)
#         # N, B, -1, c, ph, pw: 17, 1, 3, 30, 120
#         x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, c,
#                                              *padsize)  # x_h_cut_unfold.size(0), -1, padsize, padsize
#         x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
#         y_h_cut_unfold = []
#         x_h_cut_unfold.cuda()
#
#         for i in range(x_range):
#             res = self.forward(*[s[:, 0, ...] for s in
#                                  split_func(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)],
#                                **kwargs)
#             y_h_cut_unfold.append(res)
#             # y_h_cut_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
#             #                                    ...], [1, 1], dim=1)], **kwargs)])  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#
#         # [[a0, b0, c0], [a1, b1, c1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
#
#         if isinstance(y_h_cut_unfold[0], tuple):
#             y_h_cut_unfold_out = [None] * len(y_h_cut_unfold[0])
#             for i, s in enumerate(zip(*y_h_cut_unfold)):
#                 y_h_cut_unfold_out[i] = s
#             y_h_cut_unfold = y_h_cut_unfold_out
#         else:
#             y_h_cut_unfold = [y_h_cut_unfold]
#         y_h_cut = []
#
#         for s_h_cut_unfold in y_h_cut_unfold:
#             s_h_cut_unfold = torch.cat(s_h_cut_unfold, dim=0).cpu()
#             # s_h_cut_unfold = s_h_cut_unfold.reshape(-1, c, *s_h_cut_unfold.size()[1:])
#             # nH*nW, c, k, k: 3, 3, 100, 100 (17, 3, 30, 120)
#             # out_size=(30, 600), k=(30, 120)
#             s_h_cut = F.fold(
#                 s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                 (padsize[0] * scale, (w - w_cut) * scale), (padsize[0] * scale, padsize[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#             s_h_cut_unfold = s_h_cut_unfold[..., :,
#                              int(shave[1] / 2 * scale):padsize[1] * scale - int(shave[1] / 2 * scale)].contiguous()
#             # 17, 3, 30, 60
#             # out_size=(30, 540), k=(30, 90)
#             s_h_cut_inter = F.fold(
#                 s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                 (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
#                 (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#
#             s_ones = torch.ones(s_h_cut_inter.shape, dtype=s_h_cut_inter.dtype)
#             divisor = F.fold(
#                 F.unfold(s_ones, (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
#                          stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale))),
#                 (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
#                 (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#             s_h_cut_inter = s_h_cut_inter.cpu() / divisor
#
#             s_h_cut[..., :, int(shave[1] / 2 * scale):(w - w_cut) * scale - int(shave[1] / 2 * scale)] = s_h_cut_inter
#             y_h_cut.append(s_h_cut)
#
#         return y_h_cut
#
#     def cut_w(self, x_w_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):
#
#         split_func = self.split_func
#         # 1, 3, 792, 120 -> (30, 120), s=(7, 30) -> 109, 1, 3, 30, 120
#         x_w_cut_unfold = F.unfold(x_w_cut, padsize, stride=(int(shave[0] / 2), int(shave[1] / 2))).permute(2, 0,
#                                                                                                            1).contiguous()
#
#         x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, c, *padsize)
#         x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
#         y_w_cut_unfold = []
#         x_w_cut_unfold.cuda()
#
#         # TODO: [[a0, b0], [a1, b1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
#         for i in range(x_range):
#             res = self.forward(*[s[:, 0, ...] for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
#                                                                   ...], [1, 1], dim=1)], **kwargs)
#             y_w_cut_unfold.append(res)
#             # y_w_cut_unfold.append((s.cpu() for s in self.forward(*[s[:, 0, ...]for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
#             #                                    ...], [1, 1], dim=1)], **kwargs)))  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#         if isinstance(y_w_cut_unfold[0], tuple):
#             y_w_cut_unfold_out = [None] * len(y_w_cut_unfold[0])
#             for i, s in enumerate(zip(*y_w_cut_unfold)):
#                 y_w_cut_unfold_out[i] = s
#             y_w_cut_unfold = y_w_cut_unfold_out
#         else:
#             y_w_cut_unfold = [y_w_cut_unfold]
#
#         y_w_cut = []
#         for s_w_cut_unfold in y_w_cut_unfold:
#             s_w_cut_unfold = torch.cat(s_w_cut_unfold, dim=0).cpu()
#             # s_w_cut_unfold = s_w_cut_unfold.reshape(-1, c, *s_w_cut_unfold.size()[1:])
#             # 109,3,30,120
#             # out_size=(786, 120), k=(30, 120)
#             s_w_cut = F.fold(
#                 s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                 ((h - h_cut) * scale, padsize[1] * scale), (padsize[0] * scale, padsize[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#             s_w_cut_unfold = s_w_cut_unfold[...,
#                              int(shave[0] / 2 * scale):padsize[0] * scale - int(shave[0] / 2 * scale),
#                              :].contiguous()
#             # 109, 3, 16, 120
#             # out_size=(771, 120), k=(15, 120)
#             s_w_cut_inter = F.fold(
#                 s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#                 ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
#                 (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#
#             s_ones = torch.ones(s_w_cut_inter.shape, dtype=s_w_cut_inter.dtype)
#             divisor = F.fold(
#                 F.unfold(s_ones, (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
#                          stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale))),
#                 ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
#                 (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
#                 stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
#             s_w_cut_inter = s_w_cut_inter.cpu() / divisor
#
#             s_w_cut[..., int(shave[0] / 2 * scale):(h - h_cut) * scale - int(shave[0] / 2 * scale), :] = s_w_cut_inter
#             y_w_cut.append(s_w_cut)
#
#         # return y_w_cut


def crop_inference(model: BaseModel,
                   xs: Tuple[Tensor, Tensor, Tensor],
                   crop_size: Tuple[int] = (16, 64, 64),
                   stride: Tuple[int] = (8, 32, 32)):
    # only support CAVE dataset
    # input shape: 128, 512, 512

    # xs: (hsi_lr, hsi_up, rgb)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

    # preprocessing
    crop_xs = []
    ncols = []
    bs, out_c, _, _ = xs[0].shape
    _, _, out_h, out_w = xs[-1].shape
    for i in range(len(xs)):
        x = xs[i]
        _, c, h, _ = x.shape  # assume h equals w
        crop = crop_size[i]
        s = stride[i]

        ncol = (h - crop) // s
        ncols.append(ncol)
        crop_x = F.unfold(x, crop, stride=s)
        crop_x = einops.rearrange(crop_x, 'b (c k l) m -> m b c k l', k=crop, l=crop, c=c)
        crop_xs.append(crop_x)

    # model inference
    model.eval()
    out = []
    for i in range(crop_xs[0].size(0)):
        input = [crop_xs[j][i].cuda(0) for j in range(len(xs))]
        out.append(model.val_step(*input).detach().cpu())  # [bs * 225, 31, 64, 64]
        del input
        torch.cuda.empty_cache()
    # input: 255*[b, 31, 64, 64]
    out = torch.cat(out, dim=0)

    # postprocessing
    out = einops.rearrange(out, '(m b) c k l -> b (c k l) m', b=bs, k=crop_size[-1], l=crop_size[-1], c=out_c)
    output = F.fold(out, output_size=(out_h, out_w),
                    kernel_size=(crop_size[-1], crop_size[-1]),
                    dilation=1,
                    padding=0,
                    stride=(stride[-1], stride[-1]))

    # ncol = ncols[-1]
    # out = out.view(bs, -1, out_c, crop_size[-2], crop_size[-1])  # [bs, 225, 64, 64]
    # output = torch.zeros(bs, out_c, out_h, out_w)
    # for bi in range(bs):
    #     for i in range(ncol):
    #         for j in range(ncol):
    #             y = out[bi]  # [255, 64, 64]

    return output


if __name__ == '__main__':
    from model.dcformer_reduce import DCFormer_Reduce

    model = DCFormer_Reduce(8, 'C').cuda(0)

    ms = torch.randn(1, 8, 128, 128)
    interp_ms = F.interpolate(ms, size=512)

    lms = torch.randn(1, 8, 512, 512)
    pan = torch.randn(1, 1, 512, 512)
    expand_pan = pan.expand(-1, 8, -1, -1)

    # print(model.val_step(ms, lms, pan).shape)

    print(crop_inference(model, xs=(ms, lms, pan)).shape)
