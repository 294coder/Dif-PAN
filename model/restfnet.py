# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/5/21 18:43
# @Author  : Xiao Wu
# @reference: 
#
import math

import torch
import torch.nn as nn

import sys
sys.path.append('./')

from model.base_model import register_model, BaseModel


@register_model('restfnet')
class ResTFNet(BaseModel):
    def __init__(self,
                 is_bn,
                 n_select_bands,
                 n_bands,
                 scale_ratio=None,):
        """Load the pretrained ResNet and replace top fc layer."""
        super(ResTFNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands


        self.lr_conv1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
        )
        self.lr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.lr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.hr_conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
        )
        self.hr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.hr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.fusion_conv3 = nn.Sequential(
            nn.BatchNorm2d(256) if is_bn else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(256) if is_bn else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.recons_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
        )
        self.recons_conv2 = nn.Sequential(
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.recons_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.recons_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
        )
        self.recons_conv5 = nn.Sequential(
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.recons_conv6 = nn.Sequential(
            nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _forward_implem(self, x_lr, x_hr):
        # feature extraction
        # x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = x_lr + x_lr_cat
        x_lr = self.lr_down_conv(x_lr)

        x_hr = self.hr_conv1(x_hr)
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = x_hr + x_hr_cat
        x_hr = self.hr_down_conv(x_hr)
        x = torch.cat((x_hr, x_lr), dim=1)

        # feature fusion
        x = x + self.fusion_conv1(x)
        x_fus_cat = x
        x = self.fusion_conv2(x)
        x = x + self.fusion_conv3(x)
        x = self.fusion_conv4(x)
        x = torch.cat((x_fus_cat, x), dim=1)

        # image reconstruction
        x = self.recons_conv1(x)
        x = x + self.recons_conv2(x)
        x = self.recons_conv3(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv4(x)

        x = x + self.recons_conv5(x)
        x = self.recons_conv6(x)

        return x # , 0, 0, 0, 0, 0
    def train_step(self, lrhsi, up_hs, rgb, gt, criterion):
        # log_vars = {}
        sr = self._forward_implem(up_hs, rgb)
        loss = criterion(sr, gt)
        # outputs = loss
        # return loss
        # log_vars.update(loss=loss.item())
        # metrics = {'loss': loss, 'log_vars': log_vars}
        return sr, loss

    def val_step(self, lrhsi, up_hs, rgb):
        # gt, lms, ms, pan = dat
        sr = self._forward_implem(up_hs, rgb)

        return sr

# class TFNet(nn.Module):
#     def __init__(self,
#                  n_select_bands,
#                  n_bands,
#                  scale_ratio=None):
#         """Load the pretrained ResNet and replace top fc layer."""
#         super(TFNet, self).__init__()
#         self.scale_ratio = scale_ratio
#         self.n_bands = n_bands
#         self.n_select_bands = n_select_bands

#         self.lr_conv1 = nn.Sequential(
#             nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#         )
#         self.lr_conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  #
#             nn.PReLU(),
#         )
#         self.lr_down_conv = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
#             # nn.PReLU(),
#         )
#         self.hr_conv1 = nn.Sequential(
#             nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#         )
#         self.hr_conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  #
#             nn.PReLU(),
#         )
#         self.hr_down_conv = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
#             # nn.PReLU(),
#         )

#         self.fusion_conv1 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#         )
#         self.fusion_conv2 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
#             nn.PReLU(),
#         )

#         self.recons_conv1 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  #
#             nn.PReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
#             nn.PReLU(),
#         )
#         self.recons_conv2 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  #
#             nn.PReLU(),
#             nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
#             # nn.PReLU(),
#             nn.Tanh(),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x_lr, x_hr):
#         # feature extraction
#         # x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
#         x_lr = self.lr_conv1(x_lr)
#         x_lr_cat = self.lr_conv2(x_lr)
#         x_lr = self.lr_down_conv(x_lr_cat)  #

#         x_hr = self.hr_conv1(x_hr)
#         x_hr_cat = self.hr_conv2(x_hr)
#         x_hr = self.hr_down_conv(x_hr_cat)
#         x = torch.cat((x_hr, x_lr), dim=1)

#         # feature fusion
#         x = self.fusion_conv1(x)  #
#         x = torch.cat((x, self.fusion_conv2(x)), dim=1)

#         # image reconstruction
#         x = self.recons_conv1(x)
#         x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
#         x = self.recons_conv2(x)  #

#         return x#, 0, 0, 0, 0, 0

#     def train_step(self, data, *args, **kwargs):
#         # log_vars = {}
#         gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
#                            data['lrhsi'].cuda(), data['rgb'].cuda()
#         sr = self(up_hs, rgb)
#         loss = self.criterion(sr, gt, *args, **kwargs)['loss']
#         # outputs = loss
#         # return loss
#         # log_vars.update(loss=loss.item())
#         # metrics = {'loss': loss, 'log_vars': log_vars}
#         return sr, loss

#     def val_step(self, data, *args, **kwargs):
#         # gt, lms, ms, pan = data
#         gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
#                            data['lrhsi'].cuda(), data['rgb'].cuda()
#         sr = self(up_hs, rgb)

#         return sr, gt


if __name__ == '__main__':
    device = 'cuda:0'
    
    net = ResTFNet(True, 3, 50).to(device)
    
    ms = torch.randn(1, 50, 4, 4).to(device)
    lms = torch.randn(1, 50, 80, 80).to(device)
    rgb = torch.randn(1, 3, 80, 80).to(device)
    
    print(net._forward_implem(lms, rgb).shape)