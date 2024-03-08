from functools import partial
from typing import Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from model.base_model import BaseModel, register_model
from model.module.attention import (
    Attention,
    CrossAttention,
    CAttention,
    GAU,
    SGA,
    DynamicAttentionConv,
    ReducedAttention,
    MultiScaleWindowCrossAttention,
    NAFBlock,
    SCAM,
    SpatialIncepAttention,
    SwinTransformerBlock,
)
from model.module.mobile_vit_v2_attn import MobileViTBlock
from model.module.layer_norm import LayerNorm
from model.module.mlp import Mlp, Mlp2d
from model.module.resblock import Resblock, MultiScaleEncoder
from model.module.stem import ViTStem, OverlapPatchEmbed


def pre_unpatchify(x, h, w):
    ntoken = x.shape[1]
    assert h * w == ntoken, "size unmatch"
    x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
    return x


def patchify(x):
    return einops.rearrange(x, "b c h w -> b (h w) c")


class PreUnpatchify(nn.Module):
    def __init__(self, fn, h, w, **kwargs):
        super(PreUnpatchify, self).__init__()
        self.fn = fn
        self.h = h
        self.w = w
        self.fn_kwargs = kwargs

    def forward(self, x):
        x = pre_unpatchify(x, self.h, self.w)
        return patchify(self.fn(x, **self.fn_kwargs))


class PanFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: Union[int, Tuple[int, int]],
        heads: int = 4,
        attn_drop: float = 0.0,
        mlp_ratio: int = 2,
        mlp_act: nn.Module = nn.GELU,
        norm_layer=partial(LayerNorm, LayerNorm_type="BiasFree"),
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        attn_type: str = "self",
        ffn_type: str = "2d",
        bias: bool = True,
        scale_factor: int = None,
        window_size: Union[int, Tuple[int, int]] = (None, None),
        reduce_factor: int = 4,
        input_res: int = None,
        shift_size: int = 0,
    ):
        """
            used in DCFormer arch
        """
        super(PanFormerEncoderLayer, self).__init__()
        if isinstance(dim, int):
            dim_head = dim // heads
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        if attn_type == "self":  # self-attention
            self.attn = Attention(dim, heads, dim_head, dropout=attn_drop)
        elif attn_type == "cross":  # cross-attention for different q, k, v
            self.attn = CrossAttention(
                dim[0], dim[1], heads, False, attn_drop=attn_drop
            )
        elif attn_type == "C":  # Channel Attention from Restormer
            self.attn = CAttention(dim, heads, bias=False, attn_drop=attn_drop)  # bias is True in ablation
        elif attn_type == "S":  # stop gradient attention
            self.attn = SGA(dim, heads, bias, attn_drop=attn_drop)
        elif attn_type == "D":  # kernel generation (spectrum injection)
            self.attn = DynamicAttentionConv(dim, heads, attn_drop=attn_drop)
        elif attn_type == "R":  # token reduction
            self.attn = ReducedAttention(
                dim[0],
                dim[1],
                heads,
                bias=bias,
                attn_drop=attn_drop,
                reduce_ratio=reduce_factor,
            )
        elif attn_type == "M":  # multi-scale window attention
            self.attn = MultiScaleWindowCrossAttention(
                dim[0],
                dim[1],
                window_size[0],
                window_size[1],
                num_heads=heads,
                bias=True,
                attn_drop=attn_drop,
            )
            # ffn_type = None  # no ffn in ablation
        elif attn_type == "SW":
            # contains pe
            self.attn = SwinTransformerBlock(
                dim,
                input_res,
                heads,
                window_size,
                shift_size,
                mlp_ratio,
                bias,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
            ffn_type = None
        elif attn_type == "N":  # simple gating block (similar to GELU)
            self.attn = NAFBlock(dim, FFN_Expand=mlp_ratio, drop_out_rate=attn_drop)
            ffn_type = None
        elif attn_type == "SCAM":  # cross gating
            self.attn = SCAM(dim[0], dim[1])
        elif attn_type == "I":  # vgg like attention (3x3, 1x3, 3x1)
            self.attn = SpatialIncepAttention(
                dim[0],
                dim[1],
                num_heads=heads,
                embed_dim=dim[0],
                qkv_bias=bias,
                attn_drop=attn_drop,
                proj_drop=mlp_drop,
            )
        elif attn_type == "Mo":
            self.attn = MobileViTBlock(
                dim[0],
                dim[1],
                attn_dim=dim[0],
                ffn_mul=2,
                n_attn_block=2,
                attn_dropout=attn_drop,
                ffn_dropout=mlp_drop,
                patch_h=8,
                patch_w=8,
            )
            ffn_type = None

        else:
            raise NotImplementedError

        dim = dim[0] if isinstance(dim, (tuple, list)) else dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if ffn_type == "1d":  # for 1d(sequence) input
            self.ffn = Mlp(dim, int(mlp_ratio * dim), act_layer=mlp_act, drop=mlp_drop)
        elif ffn_type == "2d":  # for 2d(image) input
            self.ffn = Mlp2d(dim, mlp_ratio, bias=False)
        elif ffn_type is None:
            # if you define ffn already in just a block with attn,
            # set @ffn_type to None
            # act as a placeholder
            self.ffn = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, x2=None):
        short_cut = x

        # attn
        if x2 is not None:
            # print('cross:', x.shape, x2.shape)
            x = self.attn(x, x2)
        else:
            # print('attn:', x.shape)
            x = self.attn(x)
        x = short_cut + self.drop_path(self.norm1(x))

        # ffn
        if self.ffn is not None:
            x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class PanFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        attn_drop,
        mlp_ratio,
        mlp_act=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_drop=0.0,
        drop_path=0.0,
    ):
        super(PanFormerDecoderLayer, self).__init__()
        dim_head = dim // heads
        self.attn = Attention(dim, heads, dim_head, dropout=attn_drop)
        self.cross_attn = CrossAttention(dim, heads, dim_head, dropout=attn_drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.ffn = Mlp(dim, int(mlp_ratio * dim), act_layer=mlp_act, drop=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, tgt, mem):
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(tgt2)
        tgt = tgt + self.drop_path(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, mem)
        tgt = tgt + self.drop_path(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.drop_path(tgt2)

        return tgt


# single scale
# naive implementation
@register_model("panformer_single_scale")
class PanFormer(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c,
        out_c,
        patch_size=(2, 2),
        ms_size=(16, 16),
        pan_size=(64, 64),
    ):
        super(PanFormer, self).__init__()
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.patch_size = patch_size
        self.ms_size = ms_size
        self.pan_size = pan_size
        self.ms_ntoken = (ms_size[0] // patch_size[0]) * (ms_size[1] // patch_size[1])
        self.pan_ntoken = (pan_size[0] // patch_size[0]) * (
            pan_size[1] // patch_size[1]
        )

        self.stem1 = ViTStem(hidden_c, hidden_c, patch_size=patch_size)
        # self.stem2 = ViTStem(in_c, hidden_c, patch_size=patch_size)
        self.pe_pan = nn.Parameter(torch.zeros(1, self.pan_ntoken, hidden_c))
        # self.pe_mem_up = nn.Upsample(self.pan_ntoken)

        self.backbone = nn.Sequential(
            Resblock(hidden_c),
            Resblock(hidden_c),
            Resblock(hidden_c),
            Resblock(hidden_c),
        )

        # ms: 16*16/4/4 = 16
        # pan: 64*64/4/4 = 256
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c, 2, 0.0, 4, drop_path=0.0, mlp_drop=0.0, ffn_type="1d"
                ),
                PanFormerEncoderLayer(
                    hidden_c, 4, 0.0, 4, drop_path=0.0, mlp_drop=0.0, ffn_type="1d"
                ),
                PanFormerEncoderLayer(
                    hidden_c, 4, 0.0, 4, drop_path=0.0, mlp_drop=0.0, ffn_type="1d"
                ),
                PanFormerEncoderLayer(
                    hidden_c, 8, 0.0, 4, drop_path=0.0, mlp_drop=0.0, ffn_type="1d"
                ),
            ]
        )
        # self.linear = nn.Linear(hidden_c, out_c)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=in_c,
            kernel_size=8,
            stride=4,
            padding=2,
            bias=True,
        )
        self.deconv_out = nn.ConvTranspose2d(
            in_channels=hidden_c,
            out_channels=hidden_c,
            kernel_size=8,
            stride=4,
            padding=2,
            bias=True,
        )
        self.conv1 = nn.Conv2d(in_c, hidden_c, 1, 1)
        self.conv_out = nn.Conv2d(hidden_c, out_c, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        # hms = self.deconv(ms)
        pan_concat = pan.repeat(1, self.in_c, 1, 1)
        x = pan_concat - lms

        x = self.conv1(x)
        x = self.backbone(x)

        x = self.stem1(x) + self.pe_pan

        for layer in self.transformer:
            x = layer(x)
        # x = self.linear(x)  # [:, :self.ms_ntoken, :]
        x = x.view(
            -1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c
        ).permute(0, -1, 1, 2)
        x = self.deconv_out(x)

        x = self.conv_out(x)
        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


# multi-scale to boost performance
# keep ms, mms and lms will cause higher memory usage
@register_model("panformer_Unet_multi_scale")
class PanFormerUNet(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=64,
        multi_channels=(12, 24, 32),
        nhead=(2, 4, 8, 2),
        attn_drop=(0.2, 0.2, 0.0, 0.0),
        drop_path=(0.1, 0.1, 0.0, 0.0),
        mlp_ratio=(2, 2, 4, 4),
        mlp_drop=(0.2, 0.2, 0.0, 0.0),
    ):
        super(PanFormerUNet, self).__init__()
        self.encoder = MultiScaleEncoder(in_c, multi_channels)
        self.patch_size = (4, 4)
        self.stem1 = ViTStem(multi_channels[0], hidden_c, patch_size=(4, 4))
        self.stem2 = ViTStem(multi_channels[1], hidden_c, patch_size=(4, 4))
        self.stem3 = ViTStem(multi_channels[2], hidden_c, patch_size=(4, 4))
        self.ntoken = int((16 / 4) ** 2 + (32 / 4) ** 2 + (64 / 4) ** 2)
        self.n3 = int((16 / 4) ** 2)
        self.n2 = int((32 / 4) ** 2)
        self.n1 = int((64 / 4) ** 2)
        self.pe1 = nn.Parameter(torch.zeros(1, self.n1, hidden_c))
        self.pe2 = nn.Parameter(torch.zeros(1, self.n2, hidden_c))
        self.pe3 = nn.Parameter(torch.zeros(1, self.n3, hidden_c))
        for i in [self.pe1, self.pe2, self.pe3]:
            trunc_normal_(i, std=0.02)

        self.hidden_c = hidden_c
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c,
                    nhead[i],
                    attn_drop[i],
                    mlp_ratio[i],
                    drop_path=drop_path[i],
                    mlp_drop=mlp_drop[i],
                    ffn_type="1d",
                )
                for i in range(len(nhead))
            ]
        )
        # self.transformer = nn.ModuleList([
        #     PanFormerEncoderLayer(hidden_c, 2, 0.2, 2, drop_path=0.1, mlp_drop=0.2),
        #     PanFormerEncoderLayer(hidden_c, 4, 0.2, 2, drop_path=0.1, mlp_drop=0.2),
        #     PanFormerEncoderLayer(hidden_c, 8, 0., 4, drop_path=0., mlp_drop=0.),
        #     PanFormerEncoderLayer(hidden_c, 2, 0., 4, drop_path=0., mlp_drop=0.)
        # ])

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c * 3, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )
        self.pix_shuffle = nn.ModuleList()
        for i in range(2):
            self.pix_shuffle.append(
                nn.Sequential(
                    nn.Conv2d(hidden_c, hidden_c * 4, 3, 1, 1), nn.PixelShuffle(2)
                )
            )
        # self.deconv = nn.ConvTranspose2d(in_channels=self.hidden_c * 3, out_channels=hidden_c, kernel_size=8, stride=4,
        #                               padding=2, bias=True)
        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.encoder(lms, pan)
        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)
        r1 = r1 + self.pe1
        r2 = r2 + self.pe2
        r3 = r3 + self.pe3
        x = torch.cat([r1, r2, r3], dim=1)

        for layer in self.transformer:
            x = layer(x)
        x1 = (
            x[:, : self.n1, :]
            .view(-1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c)
            .permute(0, -1, 1, 2)
        )
        x2 = (
            x[:, self.n1 : self.n1 + self.n2, :]
            .view(
                -1,
                H // self.patch_size[0] // 2,
                W // self.patch_size[1] // 2,
                self.hidden_c,
            )
            .permute(0, -1, 1, 2)
        )
        x3 = (
            x[:, -self.n3 :, :]
            .view(
                -1,
                H // self.patch_size[0] // 4,
                W // self.patch_size[1] // 4,
                self.hidden_c,
            )
            .permute(0, -1, 1, 2)
        )
        x2 = F.interpolate(x2, scale_factor=2)
        x3 = F.interpolate(x3, scale_factor=4)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.pre_conv(x)
        for i in range(2):
            x = self.pix_shuffle[i](x)

        x = self.conv_out(x)
        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


# Switch q, k, v in the backbone, can exchange information from ms, mms and lms
@register_model("panformer_switch_qkv")
class PanFormerSwitch(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=128,
        multi_channels=(12, 24, 32),
        nhead=(2, 4, 8, 8),
        attn_drop=(0.0, 0.2, 0.2, 0.2),
        drop_path=(0.1, 0.2, 0.4, 0.4),
        mlp_ratio=(2, 2, 4, 4),
        mlp_drop=(0.1, 0.2, 0.4, 0.4),
    ):
        super(PanFormerSwitch, self).__init__()
        self.patch_size = (4, 4)
        self.hidden_c = hidden_c

        self.multi_scale_encoder = MultiScaleEncoder(in_c, multi_channels)
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c,
                    nhead[i],
                    attn_drop[i],
                    mlp_ratio[i],
                    drop_path=drop_path[i],
                    mlp_drop=mlp_drop[i],
                    attn_type="cross",
                    ffn_type="1d",
                )
                for i in range(len(nhead))
            ]
        )
        self.stem1 = ViTStem(multi_channels[0], hidden_c, patch_size=(4, 4))
        self.stem2 = ViTStem(multi_channels[1], hidden_c, patch_size=(4, 4))
        self.stem3 = ViTStem(multi_channels[2], hidden_c, patch_size=(4, 4))

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )
        # self.pix_shuffle = nn.ModuleList()
        # for i in range(2):
        #     self.pix_shuffle.append(nn.Sequential(
        #         nn.Conv2d(hidden_c, hidden_c * 4, 3, 1, 1),
        #         nn.PixelShuffle(2)
        #     ))

        self.de_conv = nn.ConvTranspose2d(
            in_channels=hidden_c,
            out_channels=hidden_c,
            kernel_size=8,
            stride=4,
            padding=2,
            bias=True,
        )
        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.multi_scale_encoder(lms, pan)

        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)
        # shot_cut = r1

        t0, t1, t2, t3 = self.transformer
        r2 = t0(r2, r1)
        r3 = t1(r3, r2)
        r2 = t2(r2, r3)
        r1 = t3(r1, r2)

        r1 = r1.view(
            -1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c
        ).permute(0, -1, 1, 2)
        r1 = self.pre_conv(r1)
        r1 = self.de_conv(r1)
        # for i in range(2):
        #     r1 = self.pix_shuffle[i](r1)

        r1 = self.conv_out(r1)

        return r1

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


# Restormer Attention
# do attention on Channel axis, keep CxC attention map
# but reach higher FLOPs

# gated FFN can help network do better

# TODO: experiments need to be done on other datasets: quickbird, gaofen, wv2, wv4.


@register_model("panformer_restormer")
class PanFormerUNet2(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=128,
        multi_channels=(12, 24, 32),
        nhead=(2, 4, 4, 12),
        attn_drop=(0.0, 0.2, 0.2, 0.2),
        drop_path=(0.1, 0.2, 0.2, 0.2),
        mlp_ratio=(2, 2, 4, 4),
        mlp_drop=(0.1, 0.2, 0.2, 0.2),
    ):
        super(PanFormerUNet2, self).__init__()
        # self.patch_size = (4, 4)
        self.hidden_c = hidden_c

        self.multi_scale_encoder = MultiScaleEncoder(in_c, multi_channels)
        # self.ntoken = int(16 ** 2 + 32 ** 2 + 64 ** 2)
        # self.n3 = int(16 ** 2)
        # self.n2 = int(32 ** 2)
        # self.n1 = int(64 ** 2)
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c * 3,
                    nhead[i],
                    attn_drop[i],
                    mlp_ratio[i],
                    drop_path=drop_path[i],
                    mlp_drop=mlp_drop[i],
                    attn_type="C",
                    norm_layer=partial(LayerNorm, LayerNorm_type="BiasFree"),
                    ffn_type="2d",
                )
                for i in range(len(nhead))
            ]
        )
        self.stem1 = OverlapPatchEmbed(multi_channels[0], hidden_c)
        self.stem2 = OverlapPatchEmbed(multi_channels[1], hidden_c)
        self.stem3 = OverlapPatchEmbed(multi_channels[2], hidden_c)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c * 3, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )
        # self.pix_shuffle = nn.ModuleList()
        # for i in range(2):
        #     self.pix_shuffle.append(nn.Sequential(
        #         nn.Conv2d(hidden_c, hidden_c * 4, 3, 1, 1),
        #         nn.PixelShuffle(2)
        #     ))

        self.up3_2 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )
        self.up2_1 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

        # self.de_conv = nn.ConvTranspose2d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=8, stride=4,
        #                                  padding=2, bias=True)
        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.multi_scale_encoder(lms, pan)

        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)

        r2 = self.up2_1(r2)
        r3 = self.up3_2(r3)
        x = torch.cat([r1, r2, r3], dim=1)

        for layer in self.transformer:
            x = layer(x)
        # x = x[:, :self.hidden_c]
        # x2 = x[:, self.n1:self.n1 + self.n2, :].view(-1, H, W, self.hidden_c).permute(0, -1, 1, 2)
        # x3 = x[:, -self.n3:, :].view(-1, H, W // self.patch_size[1] // 4,
        #                              self.hidden_c).permute(0, -1, 1, 2)

        # r1 = r1.view(-1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c).permute(0, -1, 1, 2)
        x = self.pre_conv(x)
        # x = self.de_conv(x)
        # for i in range(2):
        #     x = self.pix_shuffle[i](x)

        x = self.conv_out(x)

        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


@register_model("panformer_gau")
class PanFormerGAU(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=128,
        multi_channels=(12, 24, 32),
        depth=4,
        attn_drop=(0.0, 0.2, 0.4, 0.4),
    ):
        super(PanFormerGAU, self).__init__()
        self.patch_size = (4, 4)
        self.hidden_c = hidden_c

        self.multi_scale_encoder = MultiScaleEncoder(in_c, multi_channels)
        assert len(attn_drop) == depth
        self.transformer = nn.ModuleList(
            [
                GAU(
                    dim=hidden_c * 3,
                    query_key_dim=hidden_c * 3,
                    expansion_factor=1.0,
                    dropout=attn_drop[i],
                )
                for i in range(depth)
            ]
        )
        self.stem1 = OverlapPatchEmbed(multi_channels[0], hidden_c)
        self.stem2 = OverlapPatchEmbed(multi_channels[1], hidden_c)
        self.stem3 = OverlapPatchEmbed(multi_channels[2], hidden_c)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c * 3, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )

        self.up3_2 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )
        self.up2_1 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.multi_scale_encoder(lms, pan)

        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)

        r2 = self.up2_1(r2)
        r3 = self.up3_2(r3)
        x = torch.cat([r1, r2, r3], dim=1)
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        for layer in self.transformer:
            x = layer(x)

        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.pre_conv(x)

        x = self.conv_out(x)

        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


@register_model("panformer_sga")
class PanFormerUNetSGA(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=128,
        multi_channels=(12, 24, 32),
        nhead=(2, 4, 4, 12),
        attn_drop=(0.0, 0.2, 0.2, 0.2),
        drop_path=(0.1, 0.2, 0.2, 0.2),
        mlp_ratio=(2, 2, 4, 4),
        mlp_drop=(0.1, 0.2, 0.2, 0.2),
    ):
        super(PanFormerUNetSGA, self).__init__()
        # self.patch_size = (4, 4)
        self.hidden_c = hidden_c

        self.multi_scale_encoder = MultiScaleEncoder(in_c, multi_channels)
        # self.ntoken = int(16 ** 2 + 32 ** 2 + 64 ** 2)
        # self.n3 = int(16 ** 2)
        # self.n2 = int(32 ** 2)
        # self.n1 = int(64 ** 2)
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c * 3,
                    nhead[i],
                    attn_drop[i],
                    mlp_ratio[i],
                    drop_path=drop_path[i],
                    mlp_drop=mlp_drop[i],
                    attn_type="S",
                    norm_layer=partial(LayerNorm, LayerNorm_type="BiasFree"),
                    ffn_type="2d",
                )
                for i in range(len(nhead))
            ]
        )
        self.stem1 = OverlapPatchEmbed(multi_channels[0], hidden_c)
        self.stem2 = OverlapPatchEmbed(multi_channels[1], hidden_c)
        self.stem3 = OverlapPatchEmbed(multi_channels[2], hidden_c)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c * 3, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )
        # self.pix_shuffle = nn.ModuleList()
        # for i in range(2):
        #     self.pix_shuffle.append(nn.Sequential(
        #         nn.Conv2d(hidden_c, hidden_c * 4, 3, 1, 1),
        #         nn.PixelShuffle(2)
        #     ))

        self.up3_2 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )
        self.up2_1 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

        # self.de_conv = nn.ConvTranspose2d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=8, stride=4,
        #                                  padding=2, bias=True)
        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.multi_scale_encoder(lms, pan)

        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)

        r2 = self.up2_1(r2)
        r3 = self.up3_2(r3)
        x = torch.cat([r1, r2, r3], dim=1)

        for layer in self.transformer:
            x = layer(x)
        # x = x[:, :self.hidden_c]
        # x2 = x[:, self.n1:self.n1 + self.n2, :].view(-1, H, W, self.hidden_c).permute(0, -1, 1, 2)
        # x3 = x[:, -self.n3:, :].view(-1, H, W // self.patch_size[1] // 4,
        #                              self.hidden_c).permute(0, -1, 1, 2)

        # r1 = r1.view(-1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c).permute(0, -1, 1, 2)
        x = self.pre_conv(x)
        # x = self.de_conv(x)
        # for i in range(2):
        #     x = self.pix_shuffle[i](x)

        x = self.conv_out(x)

        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


@register_model("panformer_dynamic")
class PanFormerDynamic(BaseModel):
    def __init__(
        self,
        in_c,
        hidden_c=128,
        multi_channels=(12, 24, 32),
        nhead=(2, 4, 4, 12),
        attn_drop=(0.0, 0.2, 0.2, 0.2),
        drop_path=(0.1, 0.2, 0.2, 0.2),
        mlp_ratio=(2, 2, 4, 4),
        mlp_drop=(0.1, 0.2, 0.2, 0.2),
    ):
        super(PanFormerDynamic, self).__init__()
        # self.patch_size = (4, 4)
        self.hidden_c = hidden_c

        self.multi_scale_encoder = MultiScaleEncoder(in_c, multi_channels)
        # self.ntoken = int(16 ** 2 + 32 ** 2 + 64 ** 2)
        # self.n3 = int(16 ** 2)
        # self.n2 = int(32 ** 2)
        # self.n1 = int(64 ** 2)
        self.transformer = nn.ModuleList(
            [
                PanFormerEncoderLayer(
                    hidden_c * 3,
                    nhead[i],
                    attn_drop[i],
                    mlp_ratio[i],
                    drop_path=drop_path[i],
                    mlp_drop=mlp_drop[i],
                    attn_type="D",
                    norm_layer=partial(LayerNorm, LayerNorm_type="BiasFree"),
                    ffn_type="2d",
                )
                for i in range(len(nhead))
            ]
        )
        self.stem1 = OverlapPatchEmbed(multi_channels[0], hidden_c)
        self.stem2 = OverlapPatchEmbed(multi_channels[1], hidden_c)
        self.stem3 = OverlapPatchEmbed(multi_channels[2], hidden_c)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_c * 3, hidden_c, 3, 1, 1), nn.LeakyReLU()
        )
        # self.pix_shuffle = nn.ModuleList()
        # for i in range(2):
        #     self.pix_shuffle.append(nn.Sequential(
        #         nn.Conv2d(hidden_c, hidden_c * 4, 3, 1, 1),
        #         nn.PixelShuffle(2)
        #     ))

        self.up3_2 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )
        self.up2_1 = nn.Sequential(
            nn.Conv2d(
                hidden_c, hidden_c * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

        # self.de_conv = nn.ConvTranspose2d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=8, stride=4,
        #                                  padding=2, bias=True)
        self.conv_out = nn.Conv2d(hidden_c, 8, 3, 1, 1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        H, W = pan.shape[-2:]
        r1, r2, r3 = self.multi_scale_encoder(lms, pan)

        r1, r2, r3 = self.stem1(r1), self.stem2(r2), self.stem3(r3)

        r2 = self.up2_1(r2)
        r3 = self.up3_2(r3)
        x = torch.cat([r1, r2, r3], dim=1)

        for layer in self.transformer:
            x = layer(x)
        # x = x[:, :self.hidden_c]
        # x2 = x[:, self.n1:self.n1 + self.n2, :].view(-1, H, W, self.hidden_c).permute(0, -1, 1, 2)
        # x3 = x[:, -self.n3:, :].view(-1, H, W // self.patch_size[1] // 4,
        #                              self.hidden_c).permute(0, -1, 1, 2)

        # r1 = r1.view(-1, H // self.patch_size[0], W // self.patch_size[1], self.hidden_c).permute(0, -1, 1, 2)
        x = self.pre_conv(x)
        # x = self.de_conv(x)
        # for i in range(2):
        #     x = self.pix_shuffle[i](x)

        x = self.conv_out(x)

        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self(lms, pan)
        sr += lms
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self(lms, pan)
        sr += lms
        return sr


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    lms = torch.randn(1, 64, 16, 16)
    pan = torch.randn(1, 128, 64, 64)

    # [HW, HW] attention map
    # params: 4.715M
    # FLOPs: 17.932G
    # panformer = PanFormerGAU(8, multi_channels=(12, 24, 32))  # .cuda()

    # ----one head----
    # params: 1.335M
    # FLOPs: 4.731G
    # ================================
    # ----[1, 2, 4, 4] head----
    # params: 1.48M
    # FLOPs: 4.728G
    # panformer = PanFormerGAU(8, multi_channels=(16, 32, 64), hidden_c=64)

    # params: 2.718M
    # FLOPs: 9.522G
    # panformer = PanFormerUNet2(8, multi_channels=(16, 32, 64), hidden_c=64)

    # params: 2.412M
    # FLOPs: 8.527G
    # panformer = PanFormerUNetSGA(8, multi_channels=(16, 32, 64), hidden_c=64)

    # params: 4.052M
    # FLOPs: 14.958G
    # panformer = PanFormerDynamic(8, multi_channels=(16, 32, 64), hidden_c=64)

    module = PanFormerEncoderLayer(
        (64, 128), attn_type="I", bias=False, attn_drop=0.2, mlp_drop=0.2
    )
    # print(module(lms, pan))

    analysis = FlopCountAnalysis(module, (lms, pan))
    print(flop_count_table(analysis))
    # print(panformer(lms, pan).shape)
