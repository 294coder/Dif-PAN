from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
import math
from torch import einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import sys

sys.path.append("./")
sys.path.append("../")

ATTN_TYPE = "MAMBA_SS2D"
assert ATTN_TYPE in ["MAMBA_VIM", "MAMBA_SS2D", "RWKV"]
from model.module.rwkv_module import RWKV_ChannelMix_x051a as CMixBlock
from model.module.rwkv_module import RWKV_TimeMix_x051a as TMixBlock
from model.module.rwkv_module import Block as RKWVBlockCFirst

# from mamba_ssm import Mamba
from model.module.vmamba_module_v2 import VSSBlock
from model.base_model import BaseModel, register_model, PatchMergeModule


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.freqs = freqs

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        self.seq_len = ft_seq_len

        self.alter_seq_len(ft_seq_len, pt_seq_len)

    def alter_seq_len(self, ft_seq_len, pt_seq_len):
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, self.freqs)
        # freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t):
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = (
                t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            )
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin

    def __repr__(self):
        return (
            f"VisionRotaryEmbeddingFast(seq_len={self.seq_len}, freqs_cos={tuple(self.freqs_cos.shape)}, "
            + f"freqs_sin={tuple(self.freqs_sin.shape)})"
        )


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(
        dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = (
            nn.Sequential(act, nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# channel attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# self attention on each channel
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Linear(
            dim, hidden_dim * 3, bias=False
        )  # nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # b, c, h, w = x.shape
        # b, c, n = x.shape
        x = rearrange(x, "b d n -> b n d")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v]
        )

        # For 2D image
        # q, k, v = map(
        #     lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        # )

        # q = q * self.scale

        # sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        # attn = sim.softmax(dim=-1)
        # out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        # out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        # For 1D sequence
        q = q * self.scale

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        # v = v / n

        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        out = torch.einsum("b h d e, b h n d -> b h e n", context, q)
        out = rearrange(out, "b h e n -> b n (h e)")
        out = self.to_out(out)

        return out.transpose(1, 2)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def initialize_weights(net_l, scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity='relu')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class RWKVBlock(nn.Module):
    def __init__(self, n_embd, n_head, bias, tmix_drop, cmix_drop, n_layer, layer_id):
        super().__init__()
        self.c_fisrt_body = RKWVBlockCFirst(
            n_embd, n_head, bias, tmix_drop, cmix_drop, n_layer, layer_id
        )

    def forward(self, x):
        # if to_1d:
        # *_, h, w = x.shape
        # x = rearrange(x, "b c h w -> b (h w) c")
        x = self.c_fisrt_body(x)
        # if back_to_2d:
        # x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


# class MambaBlock(nn.Module):
#     def __init__(self, n_embd, d_state=16, d_conv=4, expand=2, norm_bias=True,
#                  ffn_expansion=2, dp_ratio=0.0) -> None:
#         super().__init__()
#         # here used bidirectional mamba
#         self.pre_norm = nn.LayerNorm(n_embd, bias=norm_bias)
#         self.attn= Mamba(n_embd, d_state, d_conv, expand)
#         self.drop_path = DropPath(dp_ratio) if dp_ratio > 0.0 else nn.Identity()
#         # self.ffn = nn.Sequential(
#         #                 nn.LayerNorm(n_embd, bias=norm_bias),
#         #                 nn.Linear(n_embd, n_embd * ffn_expansion),
#         #                 # SimpleGate(dim=-1),  # will reduce the channel
#         #                 nn.SiLU(),
#         #                 # nn.LayerNorm(n_embd * ffn_expansion, bias=norm_bias),
#         #                 nn.Linear(n_embd * ffn_expansion, n_embd),
#         #                 )
#         # self.ffn_drop_path = DropPath(dp_ratio) if dp_ratio > 0.0 else nn.Identity()
#         # self.body_beta = nn.Parameter(torch.ones(1, 1, n_embd) / 2, requires_grad=True)
#         # self.ffn_gamma = nn.Parameter(torch.ones(1, 1, n_embd) / 2, requires_grad=True)

#         self.apply(self._init_weights)

#     def forward(self, inp):
#         x = inp
#         x = self.attn(self.pre_norm(x)) + inp
#         # x = self.ffn_drop_path(self.ffn(x)) + x
#         return x

#     def _init_weights(self, m: nn.Module):
#         """
#         out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
#         no fc.weight found in the any of the model parameters
#         no nn.Embedding found in the any of the model parameters
#         so the thing is, VSSBlock initialization is useless

#         Conv2D is not intialized !!!
#         """
#         # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MambaInjectionBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        inner_chan,
        drop_path=0.0,
        mlp_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_size=8,
        d_state=32,
        dt_rank="auto",
        ssm_conv=7,
        ssm_ratio=2,
        mlp_ratio=4,
        forward_type="v2",
        use_ckpt=False,
        **mamba_kwargs,
    ):
        super().__init__()
        self.inner_chan = inner_chan
        self.window_size = window_size
        self.intro_conv = nn.Sequential(
            nn.Conv2d(in_chan, inner_chan, 1),
            nn.Conv2d(inner_chan, inner_chan, 3, 1, 1, groups=inner_chan),
        )
        self.inner_to_outter_conv = nn.Linear(2*inner_chan, inner_chan)
        # self.lerp = nn.Sequential(LayerNorm(in_chan),
        #                           nn.AdaptiveAvgPool2d(1),
        #                           nn.Conv2d(in_chan, inner_chan*2, 1),
        #                           Rearrange('b c 1 1 -> b 1 1 c'))
        self.mamba = VSSBlock(
            inner_chan,
            d_state=d_state,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            mlp_drop_rate=mlp_drop,
            ssm_conv=ssm_conv,
            ssm_dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            forward_type=forward_type,
            use_checkpoint=use_ckpt,
            **mamba_kwargs,
        )

        # v3: mamba in mamba
        self.mamba_inner_shared = VSSBlock(
            inner_chan,
            d_state=8,
            drop_path=drop_path,
            mlp_ratio=1,
            norm_layer=norm_layer,
            mlp_drop_rate=0.0,
            ssm_conv=ssm_conv,
            ssm_dt_rank=dt_rank,
            ssm_ratio=1,
            forward_type=forward_type,
            use_checkpoint=use_ckpt,
            **mamba_kwargs,
        )
        # self.norm = norm_layer(inner_chan)
        self.enhanced_factor = nn.Parameter(torch.zeros(1, 1, 1, inner_chan), requires_grad=True)

        # v1: add attn
        # self.adaptive_pool_size = (3, 3)
        # _pool_size = math.prod(self.adaptive_pool_size)
        # self.chan_attn = Attention(inner_chan, 4, inner_chan//4)  # 4 * 4 = 16 is d_state

        # v2: Film module
        # self.films = nn.ParameterDict(
        #     {'beta': nn.Parameter(torch.randn(1, 1, 1, inner_chan), requires_grad=True),
        #      'gamma': nn.Parameter(torch.randn(1, 1, 1, inner_chan), requires_grad=True)}
        # )
        # self.norm = nn.LayerNorm(inner_chan)
        # self.act = nn.GELU()
        # self.out_conv = nn.Linear(inner_chan, inner_chan)

    def forward(self, feat, cond):
        b, h, w, c = feat.shape
        cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=True)

        # intro_conv v1: on cond
        cond = self.intro_conv(cond)
        cond = cond.permute(0, 2, 3, 1)
        x = feat + cond

        # intro_conv v2: on catted feat and cond
        # cond = cond.permute(0, 2, 3, 1)
        # x = torch.cat([feat, cond], dim=-1)
        # x = self.intro_conv(x)
        # x_in = x

        # v3: mamba in mamba
        xs_local = window_partition(x, self.window_size)
        xs_local = self.mamba_inner_shared(xs_local)
        x_local = window_reverse(xs_local, self.window_size, h, w)
        # x_local = self.inner_norm(x_local)

        # enhance v1
        # x = self.mamba(self.inner_to_outter_conv(torch.cat([x_local, x], dim=-1)))
        # enhance v2
        x = self.mamba(x_local * self.enhanced_factor + x)

        # x_spe = F.adaptive_avg_pool2d(
        #     x_in.permute(0, 3, 1, 2), self.adaptive_pool_size
        # ).view(b, self.inner_chan, -1)  # [b, c, prod(adapt_pool_size)]
        # x_spe = self.chan_attn(x_spe).mean(dim=-1)[:, None, None]  # [b, 1, 1, c]
        # x = x + x_spe

        # gamma, beta = self.films['gamma'], self.films['beta']
        # x = x * (1 + gamma / gamma.norm()) + beta / beta.norm()
        # x = self.out_conv(self.act(self.norm(x))) + x
        return x


class Sequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.mods = nn.ModuleList(args)

    def __getitem__(self, idx):
        return self.mods[idx]
    
    def enc_forward(self, feat, cond):
        outp = feat
        for mod in self.mods:
            outp = mod(outp, cond)
        return outp

    def dec_forward(self, feat, cond):
        outp = feat
        outp = self.mods[0](outp)
        for mod in self.mods[1:]:
            outp = mod(outp, cond)
        return outp


class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


############# PanRWKV Model ################


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4))
            if time_emb_dim
            else None
        )

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time


class Permute(nn.Module):
    def __init__(self, mode="c_first"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "c_first":
            # b h w c -> b c h w
            return x.permute(0, 3, 1, 2)
        elif self.mode == "c_last":
            # b c h w -> b h w c
            return x.permute(0, 2, 3, 1)
        else:
            raise NotImplementedError


def down(chan):
    return nn.Sequential(
        # Rearrange('b h w c -> b c h w', h=h, w=w),
        Permute("c_first"),
        nn.Conv2d(chan, 2 * chan, 2, 2),
        # Rearrange('b c h w -> b h w c'),
        Permute("c_last"),
    )


def up(chan):
    return nn.Sequential(
        # Rearrange('b h w c -> b c h w', h=h, w=w),
        Permute("c_first"),
        # ver 1: use pixelshuffle
        # ver 2: directly upsample and half the channels
        nn.Conv2d(chan, chan // 2, 1, bias=False),
        # nn.PixelShuffle(2),
        nn.Upsample(scale_factor=2, mode="bilinear"),
        # Rearrange('b c h w -> b h w c'),
        Permute("c_last"),
    )


@register_model("panMamba")
class ConditionalNAFNet(BaseModel):
    def __init__(
        self,
        img_channel=3,
        condition_channel=3,
        out_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        ssm_convs=[],
        upscale=1,
        if_abs_pos=True,
        if_rope=False,
        pt_img_size=64,
        drop_path_rate=0.1,
        patch_merge=True,
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        self.rope = if_rope
        self.pt_img_size = pt_img_size

        if if_abs_pos:
            self.abs_pos = nn.Parameter(
                torch.randn(1, pt_img_size, pt_img_size, width), requires_grad=True
            )

        if if_rope:
            self.rope = VisionRotaryEmbeddingFast(
                chan, pt_seq_len=pt_img_size, ft_seq_len=None
            )

        self.intro = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channel,
                out_channels=width,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
            ),
            Rearrange("b c h w -> b h w c"),
        )

        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=False,
        )

        ## main body
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        depth = sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        inter_dpr = dpr

        chan = width
        n_prev_blks = 0
        # encoder
        for enc_i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                Sequential(
                    *[
                        MambaInjectionBlock(
                            condition_channel,
                            chan,
                            ssm_conv=ssm_convs[enc_i],
                            drop_path=inter_dpr[n_prev_blks + i],
                        )
                        for i in range(num)
                    ]
                )
            )
            self.downs.append(down(chan))
            chan = chan * 2
            n_prev_blks += num
            pt_img_size //= 2

        # middle layer
        self.middle_blks = Sequential(
            *[
                MambaInjectionBlock(
                    condition_channel,
                    chan,
                    ssm_conv=ssm_convs[-1],
                    drop_path=inter_dpr[n_prev_blks + i],
                )
                for i in range(num)
            ]
        )
        n_prev_blks += middle_blk_num

        # decoder
        ssm_convs = list(reversed(ssm_convs))
        for dec_i, num in enumerate(reversed(dec_blk_nums), enc_i):
            self.ups.append(up(chan))
            chan = chan // 2
            pt_img_size *= 2

            self.decoders.append(
                Sequential(
                    nn.Linear(chan * 2, chan),
                    *[
                        MambaInjectionBlock(
                            condition_channel,
                            chan,
                            ssm_conv=ssm_convs[dec_i - enc_i],
                            drop_path=inter_dpr[n_prev_blks + i],
                        )
                        for i in range(num)
                    ],
                )
            )
            n_prev_blks += num

        self.padder_size = 2 ** len(self.encoders)

        # for patch merging if the image is too large
        self.patch_merge = patch_merge

        # init
        print("============= init network =================")
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # print(type(m))
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def alter_ropes(self, ft_img_size):
        if ft_img_size != self.pt_img_size and self.rope:
            for rope in self.enc_ropes:
                rope.alter_seq_len(ft_img_size, rope.seq_len)
                ft_img_size //= 2

            self.middle_rope.alter_seq_len(ft_img_size, rope.seq_len)

            # for rope in self.middle_ropes:
            ft_img_size *= 2
            self.middle_rope.alter_seq_len(ft_img_size, rope.seq_len)

    def _forward_once(self, inp, cond):
        # inp_res = inp.clone()

        # if isinstance(time, int) or isinstance(time, float):
        #     time = torch.tensor([time]).to(inp.device)

        # x = inp - cond[:, :inp.shape[1]]
        # x = torch.cat([x, cond], dim=1)
        # x = torch.cat([inp, cond], dim=1)
        x = inp
        B, C, H, W = x.shape

        # x_hwwh = torch.cat([x.view(B, -1, L),
        #                       torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
        #                       dim=1)
        # x = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        # x = x.permute(0, 2, 1).contiguous()

        # t = self.time_mlp(time)
        # x = self.check_image_size(x)

        x = self.intro(x)
        if self.if_abs_pos:
            x = x + self.abs_pos

        # x = self.square_relu(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            # x = rope(x)
            x = encoder.enc_forward(x, cond)
            encs.append(x)
            x = down(x)

        # x = self.middle_rope(x)
        x = self.middle_blks.enc_forward(x, cond)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = torch.cat([x, enc_skip], dim=-1)
            x = decoder.dec_forward(x, cond)

        x = rearrange(x, "b h w c -> b c h w", h=H, w=W)
        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def _forward_implem(self, *args, **kwargs):
        return self._forward_once(*args, **kwargs)

    @torch.no_grad()
    def val_step(self, ms, lms, pan, patch_merge=None):
        if patch_merge is None:
            patch_merge = self.patch_merge

        if patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16, 16 * self.upscale, 16 * self.upscale],
                scale=self.upscale,
                patch_merge_step=self.patch_merge_step,
            )
            sr = _patch_merge_model.forward_chop(ms, lms, pan)[0] + lms
        else:
            self.alter_ropes(pan.shape[-1])
            sr = self._forward_implem(lms, pan) + lms
            self.alter_ropes(self.pt_img_size)

        return sr

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(lms, pan) + lms
        loss = criterion(sr, gt)

        return sr, loss

    def patch_merge_step(self, ms, lms, pan, **kwargs):
        sr = self._forward_implem(lms, pan)  # sr[:,[29,19,9]]
        return sr

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == "__main__":
    from torch.cuda import memory_summary

    
    # stable ver: 6.555M
    device = torch.device("cuda:0")
    net = ConditionalNAFNet(
        img_channel=8,
        condition_channel=1,
        out_channel=8,
        width=32,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        ssm_convs=[11, 11, 11],
        pt_img_size=64,
        if_rope=False,
        if_abs_pos=False,
        patch_merge=True,
    ).to(device)

    # net = MambaBlock(4).to(device)

    img_size = 16
    scale = 4
    chan = 8
    pan_chan = 1
    ms = torch.randn(1, chan, img_size, img_size).to(device)
    img = torch.randn(1, chan, img_size * scale, img_size * scale).to(device)
    cond = torch.randn(1, pan_chan, img_size * scale, img_size * scale).to(device)
    gt = torch.randn(1, chan, img_size * scale, img_size * scale).to(device)

    # net = torch.compile(net)

    out = net._forward_implem(img, cond)
    loss = F.mse_loss(out, gt)
    # loss.backward()
    print(loss)
    # find unused params
    # for n, p in net.named_parameters():
    #     if p.grad is None:
    #         print(n, "has no grad")

    # out = net(img.reshape(1, 4, -1).flatten(2).transpose(1, 2))

    # print(out.shape)

    # test patch merge
    # sr = net.val_step(ms, img, cond)
    # print(sr.shape)

    # print(torch.cuda.memory_summary(device=device))

    from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table

    net.forward = net._forward_once
    print(flop_count_table(FlopCountAnalysis(net, (img, cond))))
