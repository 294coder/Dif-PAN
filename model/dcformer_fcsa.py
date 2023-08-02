# Copyright (c) ZiHan Cao, LJ Deng (UESTC-MMHCISP). All Rights Reserved.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseModel, PatchMergeModule, register_model
from model.module.attention import MultiScaleWindowCrossAttention
from model.module.layer_norm import LayerNorm, normalization
from model.panformer import PanFormerEncoderLayer

PLANES = 31


################ MODULES #####################


# BN_MOMENTUM = 0.01
class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = normalization("ln", 4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def down_layer(mode, inplanes, planes, block, stride, norm_type, spatial_size):
    if mode == "default":
        return nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            normalization(norm_type, planes * block.expansion, spatial_size),
            # nn.BatchNorm2d(planes * block.expansion),
        )

    elif mode == "pixelshuffle":
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    inplanes // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelUnshuffle(2),
            )
        )

    elif mode == "patchmerge":
        return nn.Sequential(
            PatchMerging(
                spatial_size,
                inplanes,
            )
        )


# def upsample_layer(mode, inplanes, ):
#     return nn.Sequential(nn.Conv2d(inplanes, inplanes * 4, kernel_size=3, stride=1, padding=1, bias=False),
#                          nn.PixelShuffle(2))


# nn.Sequential does not support multi-input
class MySequential(nn.Module):
    def __init__(self, *layers):
        super(MySequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args):
        outs = args
        for layer in self.layers:
            outs = layer(*outs)  # only support multi-input and multi-output
        return outs


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, 1, 1),
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
    )


# TODO: speficify imput size acoording to number of channels
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            num_heads,
            spatial_size,
            mlp_ratio=2,
            norm_type="bn",
            attn_drop=0.0,
            mlp_drop=0.0,
            drop_path=0.0,
            downsample=None,
    ):
        super(BasicBlock2, self).__init__()
        self.block = PanFormerEncoderLayer(
            inplanes,
            num_heads,
            norm_type=norm_type,
            attn_drop=attn_drop,
            mlp_ratio=mlp_ratio,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
            attn_type="C",
            ffn_type="2d",
            norm_layer=partial(normalization, "ln", spatial_size=spatial_size),
        )
        self.downsample = downsample
        if inplanes != planes:
            self.out_conv = nn.Conv2d(inplanes, planes, 1)
        else:
            self.out_conv = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.out_conv(out)
        # print(out.shape, self.block.attn.dim)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_type="bn", downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(norm_type, planes)  # LayerNorm(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(norm_type, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, spatial_size, stride=1, downsample=None, norm_type="bn"
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(norm_type, planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1),
            nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                groups=planes,
            ),
        )
        self.bn2 = normalization(norm_type, planes, spatial_size)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = normalization(norm_type, planes * self.expansion, spatial_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        # out = self.relu(out)

        return out


class ChannelFuseBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            spatial_size,
            num_heads=8,
            norm_type="bn",
    ):
        super(ChannelFuseBlock, self).__init__()
        # self.conv_up = conv3x3(31, planes, stride)  # prefusion
        self.epsilon = 1e-4
        self.rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.LeakyReLU(
            0.2,
        )
        self.res_block = BasicBlock2(
            planes,
            planes,
            num_heads=num_heads,
            norm_type=norm_type,
            spatial_size=spatial_size,
        )

    def forward(self, x, y):
        # Pan + Ms 1,8,H,W + 1,8,H,W
        # x, y = inputs[0], inputs[1]

        # x-> z
        # y-> x
        # RGB

        rs_w = self.relu(self.rs_w)
        weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

        # y = self.conv_up(y)

        out = weight[0] * x + weight[1] * y
        out = self.relu(out)
        out_rs = self.res_block(out)

        return out_rs

        # return out


class HighResolutionModule(nn.Module):
    """
    高低分支交叉 前后branches数
    """

    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels_pre_layer,
            num_channels_cur_layer,
            num_channels,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop,
            drop_path,
            mlp_drop,
            fuse_method,
            spatial_size,
            multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )
        self.epsilon = 1e-4
        self.num_inchannels = num_inchannels
        self.num_channels_pre_layer = num_channels_pre_layer
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop,
            drop_path,
            mlp_drop,
        )
        self.fuse_layers = self._make_fuse_layers(
            spatial_size, num_heads, mlp_ratio, attn_drop, drop_path
        )  # Stage 的 1和2的CA
        if num_branches == 2:
            self.transition_layers = self._our_make_transition_layer(
                num_channels_pre_layer,
                num_channels_cur_layer,
                spatial_size,
                num_heads,
                mlp_ratio,
                norm_type=norm_type,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )

        self.relu = nn.LeakyReLU(0.2)

        self.fcc_w = nn.Parameter(
            torch.ones(num_branches + 1, num_branches + 1, dtype=torch.float32),
            requires_grad=True,
        )
        self.fcc_relu = nn.LeakyReLU(0.2)

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )

            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )

            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )

            raise ValueError(error_msg)

    # 构建具体的某一分支网络
    def _make_one_branch(
            self,
            branch_index,
            block,
            num_blocks,
            num_channels,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop,
            drop_path,
            mlp_drop,
            stride=1,
    ):
        layers = []
        # 加深卷积层，仅第一层牵涉到下采样
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    spatial_size=spatial_size,
                    num_heads=num_heads[branch_index],
                    mlp_ratio=mlp_ratio[branch_index],
                    norm_type=norm_type,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    drop_path=drop_path,
                )
            )

        return nn.Sequential(*layers)

    # 用于构建分支
    def _make_branches(
            self,
            num_branches,
            block,
            num_blocks,
            num_channels,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop,
            drop_path,
            mlp_drop,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    spatial_size,
                    num_heads,
                    mlp_ratio,
                    norm_type,
                    attn_drop,
                    drop_path,
                    mlp_drop,
                )
            )

        return nn.ModuleList(branches)

    def _our_make_transition_layer(
            self,
            num_channels_pre_layer,
            num_channels_cur_layer,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop,
            drop_path,
    ):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                1,
                                1,
                            ),  # TODO: normalization
                            PanFormerEncoderLayer(
                                (num_channels_cur_layer[i], num_channels_pre_layer[i]),
                                num_heads[i],
                                norm_type=norm_type,
                                attn_drop=attn_drop,
                                mlp_ratio=mlp_ratio[i],
                                attn_type="M",
                                drop_path=drop_path,
                                norm_layer=partial(
                                    normalization, "ln", spatial_size=spatial_size
                                ),
                            ),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                transfpn = []
                if i + 1 - num_branches_pre > 0:
                    transfpn.append(
                        nn.Sequential(
                            TransitionFPN(
                                len(num_channels_cur_layer),
                                spatial_size,
                                num_channels_cur_layer,
                                0,
                                num_heads=num_heads[i - 1],
                                mlp_ratio=mlp_ratio[i - 1],
                                norm_type=norm_type,
                                attn_drop=attn_drop,
                                drop_path=drop_path,
                            )
                        )
                    )

                transition_layers.append(nn.Sequential(*transfpn))
        # 获得同一个输入，不支持不同输入
        return nn.ModuleList(transition_layers)

    def _make_fuse_layers(
            self,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type,
            attn_drop=0.2,
            drop_path=0.2,
    ):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        PanFormerEncoderLayer(
                            (num_inchannels[i], num_inchannels[j]),
                            num_heads[i],
                            mlp_ratio=mlp_ratio[i],
                            attn_type="M",
                            norm_type=norm_type,
                            attn_drop=attn_drop,
                            drop_path=drop_path,
                            norm_layer=partial(
                                normalization, "ln", spatial_size=spatial_size
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    if num_branches == 2:
                        fuse_layer.append(
                            PanFormerEncoderLayer(
                                (num_inchannels[i], num_inchannels[j]),
                                num_heads[i],
                                norm_type=norm_type,
                                attn_drop=attn_drop,
                                mlp_ratio=mlp_ratio[i],
                                attn_type="M",
                                drop_path=drop_path,
                                norm_layer=partial(
                                    normalization, "ln", spatial_size=spatial_size
                                ),
                            )
                        )
                    elif num_branches == 3:
                        fuse_layer.append(
                            PanFormerEncoderLayer(
                                (num_inchannels[i], num_inchannels[j]),
                                num_heads[i],
                                norm_type,
                                attn_drop,
                                mlp_ratio[i],
                                attn_type="M",
                                drop_path=drop_path,
                                norm_layer=partial(
                                    normalization, "ln", spatial_size=spatial_size
                                ),
                            )
                        )
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, inputs):
        x, y = inputs[0], inputs[1]

        num_branches = self.num_branches
        fcc_w = self.fcc_relu(self.fcc_w)
        weight = fcc_w / (torch.sum(fcc_w, dim=0) + self.epsilon)

        if num_branches == 1:
            return [self.branches[0](x[0])]

        # BasicBlocks
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        #################################################################################
        if num_branches == 2:
            x_new_branch = self.transition_layers[num_branches]((*x, y))

        # stage2的上采样PCFT
        # print("up fusion")
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            # i>0, 0
            if num_branches == 2:
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[i], x[0])
            elif num_branches == 3:
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[i], x[0])
            for j in range(1, self.num_branches):
                weight_m = weight[i][0] if j == 1 else 1
                if i == j:
                    y = weight_m * y + weight[i][j] * x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = weight_m * y + weight[i][j] * F.interpolate(
                        self.fuse_layers[i][j](x[i], x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )  # 上采样已经不需要了，暂时保留
                else:
                    # stage3是全连接，在这里进行下采样
                    y = weight_m * y + weight[i][j] * self.fuse_layers[i][j](x[i], x[j])
            x_fuse.append(self.relu(y))

        if num_branches == 2:  # stage2的时候self.transition_layers的前2个是None,用于得到多输出
            for i in range(num_branches):
                if self.transition_layers[i] is not None:
                    if i < num_branches - 1:
                        x_fuse[i] = self.transition_layers[i](x_fuse[i])
            x_fuse.append(x_new_branch)

        return x_fuse


class TransitionFPN(nn.Module):
    def __init__(
            self,
            num_branches_after_trans,
            spatial_size,
            inchannels,
            outchannels,
            num_heads=8,
            mlp_ratio=2,
            attn_drop=0.2,
            drop_path=0.2,
            norm_type="bn",
    ):
        # input channels: (8, (8, 12), (8, 12, 16))
        # stage 2 @inchannels: 32
        # stage 2 @outchannels: 12 (mms branch)

        # stage 3 @inchannels: 12 (mms branch)
        # stage 3 @outchannels: 16 (ms branch)

        super(TransitionFPN, self).__init__()
        planes = PLANES
        self.num_branches = num_branches_after_trans  # num_branches_cur=2,3
        self.cfb_eps = 1e-4

        if self.num_branches == 2:
            self.b0_in_down = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, 1, 1),  # 256
                nn.Conv2d(outchannels, outchannels, 3, 2, 1, groups=outchannels),
                normalization(norm_type, outchannels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            # stage 2 in (mms)
            self.b2_in_up = nn.Sequential(
                nn.Conv2d(planes, outchannels, 1, 1),
                nn.Conv2d(
                    outchannels,
                    outchannels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )

            # extract CFB into TransitionFPN
            self.cfs_layers = ChannelFuseBlock(
                outchannels,
                outchannels,
                spatial_size,
                norm_type=norm_type,
                num_heads=num_heads[-1]
                if isinstance(num_heads, (list, tuple))
                else num_heads,
            )

            # CFB init
            # self.cfb_rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.cfb_relu = nn.LeakyReLU(0.2)
            # self.cfb_basic_block = BasicBlock2(
            #     outchannels,
            #     outchannels,
            #     num_heads=num_heads[-1]
            #     if isinstance(num_heads, (list, tuple))
            #     else num_heads,
            #     norm_type=norm_type,
            #     spatial_size=spatial_size,
            # )

        if self.num_branches == 3:
            # inchannels = [32, 64, 128]
            self.epsilon = 1e-4

            self.b2_in_up = nn.Sequential(
                nn.Conv2d(planes, inchannels[2], 1, 1),
                nn.Conv2d(
                    inchannels[2],
                    inchannels[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
            self.fuse_block_b0_b1 = PanFormerEncoderLayer(
                (inchannels[2], inchannels[0]),
                num_heads=num_heads,
                norm_type=norm_type,
                attn_drop=attn_drop,
                mlp_ratio=mlp_ratio,
                attn_type="M",
                drop_path=drop_path,  # (64, 32)->(32, 128)
                norm_layer=partial(normalization, "ln", spatial_size=spatial_size),
            )
            self.fuse_block_b1_b2 = PanFormerEncoderLayer(
                (inchannels[2], inchannels[1]),
                num_heads=num_heads,
                norm_type=norm_type,
                attn_drop=attn_drop,
                mlp_ratio=mlp_ratio,
                attn_type="M",
                drop_path=drop_path,  # (128, 64)->(64, 128)
                norm_layer=partial(normalization, "ln", spatial_size=spatial_size),
            )
            self.rs_w = nn.Parameter(
                torch.ones(2, dtype=torch.float32), requires_grad=True
            )
            self.relu = nn.LeakyReLU(0.2)

            # same as above
            self.cfs_layers = ChannelFuseBlock(
                inchannels[2],
                inchannels[2],
                spatial_size,
                norm_type=norm_type,
                num_heads=num_heads[-1]
                if isinstance(num_heads, (list, tuple))
                else num_heads,
            )

            # self.cfb_rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.cfb_relu = nn.LeakyReLU(0.2)
            # self.cfb_basic_block = BasicBlock2(
            #     inchannels[2],
            #     inchannels[2],
            #     num_heads=num_heads[-1]
            #     if isinstance(num_heads, (list, tuple))
            #     else num_heads,
            #     norm_type=norm_type,
            #     spatial_size=spatial_size,
            # )

    def forward(self, x):
        """
        :param x: 32x64x64 | 32x64x64 64x32x32 | 32x64x64 64x32x32 128x16x16
        :param y: 8x64x64 | 8x32x32 | 8x16x16
        :return:
        """
        num_branches = self.num_branches
        if num_branches == 2:
            # @b0_in: (32, 64, 64)
            # @b1_in: (planes, 32, 32)
            b0_in, b1_in = x[0], x[1]
            b1_in = self.b2_in_up(b1_in)

            out = self.cfs_layers(self.b0_in_down(b0_in), b1_in)
            # b0_in_down = self.b0_in_down(b0_in)
            #
            # cfb_rs_w = self.cfb_relu(self.cfb_rs_w)
            # cfb_w = cfb_rs_w / cfb_rs_w.sum(0) + self.cfb_eps
            # out = cfb_w[0] * b0_in_down + cfb_w[1] * b1_in
            # out = self.cfb_relu(out)
            # out = self.cfb_basic_block(out)

        if num_branches == 3:
            b0_in, b1_in, b2_in = x[0], x[1], x[2]

            b2_in = self.b2_in_up(b2_in)  # 31->128
            b0_b2_in = self.fuse_block_b0_b1(b2_in, b0_in)  # 128
            b1_b2_in = self.fuse_block_b1_b2(b2_in, b1_in)  # 128

            # CFB 1
            # cfb_rs_w = self.cfb_relu(self.cfb_rs_w)
            # cfb_w = cfb_rs_w / cfb_rs_w.sum(0) + self.cfb_eps
            # cfb_out = cfb_w[0] * b0_b2_in + cfb_w[1] * b2_in
            # cfb_out = self.cfb_relu(cfb_out)
            # cfb_out = self.cfb_basic_block(cfb_out)

            # CFB 2
            rs_w = self.relu(self.rs_w)
            weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)
            out = weight[0] * b1_b2_in + weight[1] * self.cfs_layers(b0_b2_in, b2_in)  # 对应Fig.4第三列上下的直线
            # out = weight[0] * b1_b2_in + weight[1] * cfb_out  # 对应Fig.4第三列上下的直线

        return out


@register_model("dcformer_mwsa_new")
class DCFormerMWSA(BaseModel):
    window_dict_train_reduce = {64: 16, 32: 8, 16: 4}
    window_dict = window_dict_train_reduce

    def __init__(
            self,
            spatial_size,
            spectral_num,
            mode="SUM",
            channel_list=(64, (32, 64), (32, 64, 128)),
            block_list=(1, (1, 1), (1, 1, 1)),  # (4, (4, 3), (4, 3, 2)),
            num_heads=(2, (2, 2), (2, 2, 2)),
            mlp_ratio=(2, (2, 2), (2, 2, 2)),
            attn_drop=0.2,
            drop_path=0.2,
            mlp_drop=0.0,
            added_c=1,
            norm_type="bn",
            residual=True,
            patch_merge_step=False,
            patch_size_list=[64, 64, 16],
            crop_batch_size=20,
            scale=8,
    ):
        super(DCFormerMWSA, self).__init__()
        self.first_fuse = mode
        self.added_c = added_c
        if mode == "SUM":
            print("repeated sum head")
            init_channel = spectral_num
        elif mode == "C":
            print("concat head")
            init_channel = spectral_num + added_c
            # init_channel = spectral_num + 3
        else:
            assert False, print("fisrt_fuse error")
        self.blocks_list = block_list
        self.channels_list = channel_list
        self.residual = residual
        print("residual: ", residual)

        ### use patchmerge model
        self.patch_merge = patch_merge_step
        self.scale = scale
        self.patch_list = patch_size_list
        assert scale in [4, 8], "model only support scale equals 4 and 8"
        if patch_merge_step:
            self._patch_merge_model = PatchMergeModule(
                # net=self,
                patch_merge_step=self.patch_merge_step,
                crop_batch_size=crop_batch_size,
                patch_size_list=patch_size_list,
                scale=scale,
            )

        NUM_MODULES = 1  # HighResolutionModule cell repeat
        ################################################################################################
        # stem net
        self.conv1 = nn.Conv2d(
            init_channel,
            channel_list[0],
            kernel_size=3,
            stride=1,
            padding=1,  # , bias=False
        )
        self.bn1 = normalization(
            norm_type, channel_list[0], spatial_size
        )  # nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[0], 1, 1),
            nn.Conv2d(
                channel_list[0],
                channel_list[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=channel_list[0],
            ),
        )
        self.bn2 = normalization(norm_type, channel_list[0], spatial_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        ################################################################################################
        # stage 1
        # convolution at early stage make training process stable
        print("construct stage 1")
        self.layer1 = self._make_layer(
            downsample_mode="default",
            block=Bottleneck,
            inplanes=channel_list[0],
            spatial_size=spatial_size,
            planes=self.channels_list[0],
            blocks=self.blocks_list[0],
            norm_type=norm_type,
        )
        stage1_out_channel = Bottleneck.expansion * self.channels_list[0]  # 4 * in_c
        print("--------------------------------")

        ################################################################################################
        # stage 2
        print("construct stage 2")
        self.Stage2_NUM_BRANCHES = 2

        num_channels = [
            self.channels_list[1][i] * BasicBlock2.expansion
            for i in range(len(self.channels_list[1]))
        ]

        self.transition1 = self._our_make_transition_layer(
            [stage1_out_channel],
            num_channels,
            spatial_size,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio[1],
            norm_type=norm_type,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        transition_num_channels = [
            self.channels_list[2][i] * BasicBlock2.expansion
            for i in range(len(self.channels_list[2]))
        ]
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES,
            num_branches=self.Stage2_NUM_BRANCHES,
            num_blocks=self.blocks_list[1],
            num_inchannels=num_channels,
            num_channels_pre_layer=self.channels_list[1],
            num_channels_cur_layer=transition_num_channels,
            num_channels=self.channels_list[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio[1],
            attn_drop=attn_drop,
            drop_path=drop_path,
            mlp_drop=mlp_drop,
            block=BasicBlock2,
            spatial_size=spatial_size,
        )
        # print(self.stage2)
        print("--------------------------------")

        ################################################################################################
        # stage 3
        # 三个分支分别卷积
        print("construct stage 3")
        self.Stage3_NUM_BRANCHES = 3
        num_channels = [
            self.channels_list[2][i] * BasicBlock2.expansion
            for i in range(len(self.channels_list[2]))
        ]

        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES,
            num_branches=self.Stage3_NUM_BRANCHES,
            num_blocks=self.blocks_list[2],
            num_inchannels=num_channels,
            num_channels_pre_layer=pre_stage_channels,
            num_channels_cur_layer=num_channels,
            num_channels=self.channels_list[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio[2],
            attn_drop=attn_drop,
            drop_path=drop_path,
            mlp_drop=mlp_drop,
            block=BasicBlock2,
            spatial_size=spatial_size,
        )
        print("--------------------------------")

        ################################################################################################
        # 保主分支，进行最终分支融合输出
        last_inp_channels = int(np.sum(pre_stage_channels))  # ms_channels
        FINAL_CONV_KERNEL = 1
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            normalization(norm_type, last_inp_channels, spatial_size),
            # nn.BatchNorm2d(last_inp_channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=spectral_num,
                kernel_size=FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if FINAL_CONV_KERNEL == 3 else 0,
            ),
        )

        # set all window_dict
        self._set_window_dict(self.window_dict)

    def _make_layer(
            self,
            downsample_mode,
            block,
            inplanes,
            planes,
            blocks,
            spatial_size,
            stride=1,
            norm_type="bn",
    ):
        # 产生Bottleneck
        # stem主分支进行通道扩展时(64->256),下采样/2
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(
            #         inplanes,
            #         planes * block.expansion,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     ),
            #     normalization(norm_type, planes * block.expansion, spatial_size),
            #     # nn.BatchNorm2d(planes * block.expansion),
            # )
            downsample = down_layer(
                downsample_mode,
                inplanes,
                planes,
                block,
                stride,
                norm_type,
                spatial_size,
            )

        layers = []
        layers.append(
            block(
                inplanes, planes, spatial_size, stride, downsample, norm_type=norm_type
            )
        )
        inplanes = planes * block.expansion
        for i in range(0, blocks):
            layers.append(block(inplanes, planes, spatial_size, norm_type=norm_type))

        return nn.Sequential(*layers)

    # 为每个分支进行构造卷积模块
    def _make_stage(
            self,
            num_modules,
            num_branches,
            num_blocks,
            num_inchannels,
            num_channels_pre_layer,
            num_channels_cur_layer,
            num_channels,
            num_heads,
            mlp_ratio,
            attn_drop,
            drop_path,
            mlp_drop,
            block,
            spatial_size,
            fuse_method="SUM",
            multi_scale_output=True,
    ):
        modules = []  # HRNet的多尺度目标检测
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels_pre_layer,
                    num_channels_cur_layer,
                    num_channels,
                    num_heads,
                    mlp_ratio,
                    attn_drop,
                    drop_path,
                    mlp_drop,
                    fuse_method,
                    spatial_size,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    # 全连接步进卷积,产生多分支
    def _make_transition_layer(
            self,
            num_channels_pre_layer,
            num_channels_cur_layer,
            spatial_size,
            norm_type="bn",
    ):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            normalization(
                                norm_type, num_channels_cur_layer[i], spatial_size
                            ),
                            # nn.BatchNorm2d(num_channels_cur_layer[i]),
                            # BatchNorm2d(
                            #     num_channels_cur_layer[i]),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(  # TODO 除了-8外怎么平衡
                                inchannels,
                                outchannels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            normalization(norm_type, outchannels, spatial_size),
                            # nn.BatchNorm2d(outchannels),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _our_make_transition_layer(
            self,
            num_channels_pre_layer,
            num_channels_cur_layer,
            spatial_size,
            num_heads,
            mlp_ratio,
            norm_type="bn",
            attn_drop=0.2,
            drop_path=0.0,
            mlp_drop=0.0,
    ):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                1,
                                1,
                            ),
                            PanFormerEncoderLayer(
                                num_channels_cur_layer[i],
                                num_heads[0],
                                norm_type=norm_type,
                                attn_drop=attn_drop,
                                mlp_ratio=mlp_ratio[0],
                                mlp_drop=mlp_drop,
                                drop_path=drop_path,
                                attn_type="C",
                                ffn_type="2d",
                                norm_layer=partial(
                                    normalization, "ln", spatial_size=spatial_size
                                ),
                            ),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                transfpn = []
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    transfpn.append(
                        nn.Sequential(
                            TransitionFPN(
                                len(num_channels_cur_layer),
                                spatial_size,
                                inchannels,
                                outchannels,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                norm_type=norm_type,
                                attn_drop=attn_drop,
                                drop_path=drop_path,
                            ),
                            normalization(norm_type, outchannels, spatial_size),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*transfpn))
        return nn.ModuleList(transition_layers)

    def _forward_implem(self, x, ly, my, sy):
        # pan, lms, mms, ms
        """
        TODO: ly三层卷后级联x再做transition_layer
        Input Image kind: Pan or ms
        x: high resolution Image is inputed in Network, shape:[N, C, 64, 64]
        y: low resolution image is used to fine-tune PAN image to produce lms,
                  shape:[N, C, base_rsl=16, base_rsl=16], (is ms)
                        [N, C, base_rsl*scale_up, base_rsl*scale_up], (is mms)
                        [N, C, base_rsl*scale_up, base_rsl*scale_upper], (is lms)
        :return: higher resolution image x, shape:[N, C, 64, 64]
        """
        if self.first_fuse == "SUM":
            x = x.repeat(1, 8, 1, 1)
            x = x + ly
        if self.first_fuse == "C":
            x = torch.cat([x, ly], dim=1)

        # stem layer, keep steady training process at early stage
        # pre-fusion units1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.Stage2_NUM_BRANCHES):
            if self.transition1[i] is not None:
                if i < self.Stage2_NUM_BRANCHES - 1:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(self.transition1[i]((x, my)))
            else:
                x_list.append(x)
        x_list = self.stage2((x_list, sy))

        x = self.stage3((x_list, None))
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode="bilinear", align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode="bilinear", align_corners=True)

        x = torch.cat([x[0], x1, x2], 1)

        x = self.last_layer(x)

        return x

    def init_weights(
            self,
            pretrained="",
    ):
        print("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def train_step(self, ms, lms, pan, gt, criterion):
        mms = F.interpolate(
            ms,
            size=(lms.size(-1) // 2, lms.size(-1) // 2),
            mode="bilinear",
            align_corners=True,
        )
        sr = self._forward_implem(pan, lms, mms, ms)
        if self.residual:
            sr = sr + lms
        loss = criterion(sr, gt)

        return sr, loss

    @torch.no_grad()
    def val_step(self, ms, lms, pan):
        p_size = pan.shape[-1]
        mms = F.interpolate(
            ms,
            size=(lms.size(-1) // 2, lms.size(-1) // 2),
            mode="bilinear",
            align_corners=True,
        )
        if not self.patch_merge:
            _base_win_sz = list(self.window_dict.values())[0]
            if self.scale == 4:
                # only work for upsample ratio 4
                window_dict_test_reduce = {
                    p_size // 2 ** i: _base_win_sz // 2 ** i for i in range(3)
                }
            else:
                # work for upsample ratio 8
                window_dict_test_reduce = (
                    {p_size // 2 ** i: _base_win_sz // 2 ** i for i in [0, 1, 3]}
                    if p_size % 5 != 0 and p_size % 2 == 0
                    else {
                        p_size // 2 ** i: 40 // 2 ** i
                        for i in [0, 1, 3]  # warning: it may cause OOM
                    }
                )

            self._set_window_dict(window_dict_test_reduce)
            # print(f"set window dict: {window_dict_test_reduce}")
            sr = self._forward_implem(pan, lms, mms, ms)
            self._set_window_dict(self.window_dict)
        else:
            # print("do patch merge step")
            win_sizes = list(self.window_dict_train_reduce.values())
            chop_list = self.patch_list
            # 16, 32, 64, 64
            self._set_window_dict(
                {
                    chop_list[0]: win_sizes[2],
                    chop_list[1]: win_sizes[1],
                    chop_list[2]: win_sizes[0],
                    chop_list[3]: win_sizes[0],
                }
            )
            sr = self._patch_merge_model.forward_chop(ms, mms, lms, pan)[0]
            self._set_window_dict(self.window_dict)

        if self.residual:
            sr = sr + lms

        return sr

    def patch_merge_step(self, ms, mms, lms, pan, **kwargs):
        sr = self._forward_implem(pan, lms, mms, ms)  # sr[:,[29,19,9]]
        return sr

    def _set_window_dict(self, new_window_dict: dict):
        """to set mwsa window_dict
        warning: this function would change the window_dict of mswa in the model
        which is extremely slow beacause it will for-loop the modules
        only used when you know what you are doing.
        """
        for m in self.modules():
            # window_dict = {}
            if isinstance(m, MultiScaleWindowCrossAttention):
                m.window_dict = new_window_dict
        # self.window_dict = new_window_dict


# class build_FCFormer(HISRModel, name='FCFormer_hyper_L'):
#     def __call__(self, cfg):
#         spectral_num = 31
#         loss = nn.L1Loss(size_average=True).cuda()  # Define the Loss function
#         loss2 = SSIM(size_average=True).cuda()
#         weight_dict = {'l1_loss': 1, 'ssim_loss': 0.1}
#         losses = {'l1_loss': loss, 'ssim_loss': loss2}
#         criterion = SetCriterion(losses, weight_dict)
#         # (64, (32, 64), (32, 64, 128)),
#         model = DCFormerMWSA(64, 31, "C", added_c=3, channel_list=(48, (48, 96), (48, 96, 192)),
#                              num_heads=(8, (8, 8), (8, 8, 8)),
#                              mlp_ratio=(2, (2, 2), (2, 2, 2)), attn_drop=0.0, drop_path=0.0,
#                              block_list=[1, [1, 1], [1, 1, 1]], norm_type="ln").cuda()
#         # model.window_dict_test_reduce = {64: 16, 32: 8, 16: 4}
#         # model.window_dict_test_reduce = {128: 16, 64: 8, 32: 4}
#         model.criterion = criterion
#         optimizer = optim.AdamW(
#             model.parameters(), lr=cfg.lr, weight_decay=0)  # optimizer 1: Adam
#         # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
#         scheduler = None

#         return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    # torch.cuda.set_device("cuda:1")

    # from model.module.attention import MultiScaleWindowCrossAttention

    # torch.cuda.set_device('cuda:1')

    net = DCFormerMWSA(
        64,
        31,
        "C",
        added_c=3,
        channel_list=(8, (8, 12), (8, 12, 16)),
        num_heads=(4, (4, 4), (4, 4, 4)),
        mlp_ratio=(1, (1, 1), (1, 1, 1)),
        attn_drop=0.0,
        drop_path=0.0,
        block_list=[2, [2, 2], [2, 2, 2]],
        norm_type="ln",
        patch_merge_step=False,
        patch_size_list=[
            16,
            32,
            64,
            64,
        ],
        scale=4,
        crop_batch_size=2,
    )  # .cuda()

    ms = torch.randn(1, 31, 16, 16)  # .cuda()
    mms = torch.randn(1, 31, 32, 32)  # .cuda()
    lms = torch.randn(1, 31, 64, 64)  # .cuda()
    pan = torch.randn(1, 3, 64, 64)  # .cuda()

    sr = net._forward_implem(pan, lms, mms, ms)
    loss = ((torch.randn(1, 31, 64, 64) - sr) ** 2).sum()
    loss.backward()

    sum_p = 0
    for p in net.parameters():
        if p.requires_grad and p.grad is not None:
            sum_p += p.numel()

    print(sum_p / 1e6)

