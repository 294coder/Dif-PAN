import math
from inspect import isfunction
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial BCHW tensors"""

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AttentiveGuide(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

    def forward(self, x, g):
        return self.norm1(x) * self.norm2(g) * x


class Upsample(nn.Module):
    def __init__(self, dim, save_fm=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.save_fm = save_fm

    def forward(self, x):
        x = self.conv(self.up(x))
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            # nn.BatchNorm2d(dim),
            nn.GroupNorm(groups, dim),
            # LayerNorm2d(dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim=None,
        dropout=0,
        norm_groups=32,
        attn_guide=False,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(Swish(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.atten_guide = AttentiveGuide(dim_out) if attn_guide else nn.Identity()

    def forward(self, x, time_emb, guidance=None):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        if exists(guidance):
            # guidance should have the same shape as h
            h = self.atten_guide(h, guidance)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        # self.norm = nn.BatchNorm2d(in_channel)
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = LayerNorm2d(in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        attn_guide=False,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim,
            dim_out,
            time_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
            attn_guide=attn_guide,
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups, nhead=8)

    def forward(self, x, time_emb, guidance=None):
        x = self.res_block(x, time_emb, guidance)
        if self.with_attn:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        self_condition=False,
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        if self_condition:
            in_channel += out_channel
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            print(f"unet init: use attn size: {now_res}")
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        time_emb_dim=time_dim,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        attn_guide=True,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    time_emb_dim=time_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    time_emb_dim=time_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            print(f"unet init: use attn size: {now_res}")
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        time_emb_dim=time_dim,
                        dropout=dropout,
                        norm_groups=norm_groups,
                        with_attn=use_attn,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )

        self._set_saved_flag = False
        self.res_blocks = res_blocks
        self.self_condition = self_condition

    def _forward_imple(
        self,
        x,
        time=None,
        cond: Union[list[torch.Tensor, list], torch.Tensor] = None,
        self_cond: Tensor = None,
    ):
        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, torch.zeros_like(x))
            x = torch.cat([self_cond, x], dim=1)

        # conditional guidance
        if cond is not None:
            if isinstance(cond, list) and len(cond) == 2:
                x = torch.cat([cond[0], x], dim=1)
                guidance = cond[1]
            else:
                x = torch.cat([cond, x], dim=1)
                guidance = None
        else:
            guidance = None

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        gi = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                # guidance inject in unet encoder
                gs = guidance[gi // self.res_blocks] if exists(guidance) else None
                x = layer(x, t, gs)
                gi += 1
                # print(f'downsample size {x.shape}')
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                # gi += 1
                # print(f'middle layer size {x.shape}')
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

    def forward(self, *args, **kwargs):
        """
        suit with dpm-solver api
        """
        if kwargs.get("interm_fm") is not None and kwargs.get("interm_fm"):
            kwargs.pop("interm_fm")
            return self.interm_fm_eval_forward(*args, **kwargs)
        else:
            return self._forward_imple(*args, **kwargs)

    def _set_upsample_saved_fm(self, saved=True):
        for m in self.modules():
            if isinstance(m, Upsample):
                m.save_fm = saved
        self._set_saved_flag = saved

    @torch.no_grad()
    def interm_fm_eval_forward(
        self,
        x,
        time,
        cond=None,
        self_cond=None,
        saved_times: list = [60, 40, 20],
        verbose=False,
    ):
        """call this function when @denoise_fn module is
        hooked(a module hook that saved intermediate feature
        map)

        Args:
            x (tensor): input
            time (tensor): time sequence
            cond (tensor, optional): condition. Defaults to None.
            saved_times (list, optional): time steps that used to save intermediate feature map.
                                          Defaults to [60, 40, 20].
            verbose (bool, optional): print saved time steps. Defaults to True.

        Returns:
            tensor: output
        """
        if (
            np.ceil(time[0].item()).astype("int") in saved_times
            and not self._set_saved_flag
        ):
            if verbose:
                print(f"get intermediate feature map at timestep {time[0].item()}")
            self._set_upsample_saved_fm(True)

        x = self._forward_imple(x, time, cond, self_cond)

        if self._set_saved_flag:
            self._set_upsample_saved_fm(False)
        self._set_saved_flag = False
        return x


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net = UNet(
        in_channel=9,
        out_channel=8,
        image_size=64,
        res_blocks=3,
        attn_res=(8, 16),
        inner_channel=64,
        channel_mults=(1, 2, 4, 4),
        self_condition=False,
        with_time_emb=False,
    )
    x = torch.randn(1, 9, 64, 64)
    t = torch.tensor([1.0])
    # gs = [
    #     torch.randn(1, 9, 64, 64),
    #     [
    #         torch.randn(1, 32, 64, 64),
    #         torch.randn(1, 64, 32, 32),
    #         torch.randn(1, 128, 16, 16),
    #         torch.randn(1, 128, 8, 8),
    #     ],
    # ]
    # cond = torch.randn(1, 9, 64, 64)
    # self_cond = torch.randn(1, 8, 64, 64)

    print(net(x).shape)

    # analysis = FlopCountAnalysis(net, (x, t, gs))
    # print(flop_count_table(analysis))

    # for n, m in net.named_modules():
    #     print(f"{n}")
