import math
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from timm.models.layers import DropPath

# from diffusion_engine import norm


def base2fourier_features(inputs, freq_start=7, freq_stop=8, step=1):
    freqs = range(freq_start, freq_stop, step)
    w = (
        2.0 ** (torch.tensor(freqs, dtype=inputs.dtype, device=inputs.device))
        * 2
        * torch.pi
    )
    w = w[None, :].repeat(1, inputs.shape[1])

    # compute features
    h = inputs.repeat_interleave(len(freqs), dim=1)
    h = w[..., None, None] * h
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
    return h


class UNetSR3(nn.Module):
    def __init__(
        self,
        in_channel=8,
        out_channel=3,
        inner_channel=32,
        lms_channel=8,
        pan_channel=1,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        self_condition=False,
        fourier_features=False,
        fourier_min=7,
        fourier_max=8,
        fourier_step=1,
        pred_var=False,
    ):
        super().__init__()

        self.lms_channel = lms_channel
        self.pan_channel = pan_channel

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        if self_condition:
            in_channel += out_channel
        if fourier_features:
            n = np.ceil((fourier_max - fourier_min) / fourier_step).astype("int")
            in_channel += in_channel * n * 2

        self.fourier_features = fourier_features
        self.fourier_min = fourier_min
        self.fourier_max = fourier_max
        self.fourier_step = fourier_step

        self.pred_var = pred_var

        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        cond_dim=lms_channel + pan_channel,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        encoder=True,
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
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
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
            if use_attn:
                print("use attn: res {}".format(now_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        encoder=False,
                        cond_dim=lms_channel + pan_channel * 3,
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

        self.res_blocks = res_blocks
        self.self_condition = self_condition
        # self.add_cond_layer_wise = add_cond_layer_wise

    def forward(self, x, time, cond=None, self_cond=None):

        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([self_cond, x], dim=1)

        # if cond is not None:
        #     x = torch.cat([cond, x], dim=1)

        if self.fourier_features:
            x = torch.cat(
                [
                    x,
                    base2fourier_features(
                        x, self.fourier_min, self.fourier_max, self.fourier_step
                    ),
                ],
                dim=1,
            )

        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []
        # TODO: 在encoder中加入cross attn
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    x, t, cond[:, : self.lms_channel + self.pan_channel]
                )  # cond: cat[lms, pan]
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    torch.cat((x, feats.pop()), dim=1),
                    t,
                    cond[:, -self.lms_channel - self.pan_channel * 3 :],
                )  # cond: cat[lms_main, pan_h, pan_v]
            else:
                x = layer(x)

        return self.final_conv(x)  # + res


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


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
            nn.GroupNorm(groups, dim),
            # nn.BatchNorm2d(dim),
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
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = torch.nn.functional.normalize
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


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class CondInjection(nn.Module):
    def __init__(self, fea_dim, cond_dim, hidden_dim, groups=32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 4, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1, bias=True),
        )
        self.x_conv = nn.Conv2d(fea_dim, hidden_dim, 1, bias=True)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x, cond):
        cond = self.body(cond)
        scale, shift = cond.chunk(2, dim=1)

        x = self.x_conv(x)

        x = x * (1 + scale) + shift
        return x


class FreqCondInjection(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0, "@dim must be divisible by @nheads"

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)
        # self.prenorm_cond = nn.GroupNorm(groups // 4, cond_dim)

        self.q = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim, 1, bias=True),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, 1, 1, bias=False, groups=cond_dim),
            nn.Conv2d(cond_dim, qkv_dim * 2, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, cond):
        x = self.prenorm_x(x)
        # cond = self.prenorm_cond(cond)

        q = self.q(x)
        k, v = self.kv(cond).chunk(2, dim=1)

        q, k, v = map(lambda in_qkv: F.normalize(in_qkv, dim=1), (q, k, v))

        # convert to freq space
        q = torch.fft.rfft2(q, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        k = torch.fft.rfft2(k, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        v = torch.fft.rfft2(v, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1

        # amp and phas attention
        amp_out = self.attn_op(q.abs(), k.abs(), v.abs())
        phas_out = self.attn_op(q.angle(), k.angle(), v.angle())

        # convert to complex
        out = torch.polar(amp_out, phas_out)

        # convert to rgb space
        out = torch.fft.irfft2(out, dim=(-2, -1), norm="ortho")

        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out

    def attn_op(self, q, k, v):
        b, c, xf, yf = q.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        # n x n attn map
        sim = torch.einsum("b h c m, b h c n -> b h m n", q, k) * self.scale
        sim = sim.softmax(-1)
        # h w fused feature map
        out = torch.einsum("b h m n, b h c n-> b h c m", sim, v)
        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )

        return out


class FastAttnCondInjection(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0, "@dim must be divisible by @nheads"

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)
        # self.prenorm_cond = nn.GroupNorm(groups // 4, cond_dim)

        self.q = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim, 1, bias=True),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, 1, 1, bias=False, groups=cond_dim),
            nn.Conv2d(cond_dim, qkv_dim * 2, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, cond):
        x = self.prenorm_x(x)
        # cond = self.prenorm_cond(cond)

        q = self.q(x)
        k, v = self.kv(cond).chunk(2, dim=1)

        # q, k, v = map(lambda in_qkv: F.normalize(in_qkv, dim=1), (q, k, v))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        # convert to freq space
        # q = torch.fft.rfft2(q, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        # k = torch.fft.rfft2(k, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        # v = torch.fft.rfft2(v, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1

        b, c, xf, yf = q.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        q = q * self.scale
        # c x c attn map
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # h w fused feature map
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )

        # convert to rgb space
        # out = torch.fft.irfft2(out, dim=(-2, -1), norm="ortho")

        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out


class WrappedCondInj(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        ffn_drop_path=0.2,
    ) -> None:
        super().__init__()
        self.rgb_cond_inj = CondInjection(fea_dim, cond_dim, dim_out, groups=groups)
        self.fft_cond_inj = FastAttnCondInjection(
            fea_dim,
            cond_dim,
            qkv_dim,
            dim_out,
            groups=groups,
            nheads=nheads,
            drop_path_prob=ffn_drop_path,
        )
        self.to_out = nn.Conv2d(dim_out * 2, dim_out, 1, bias=True)

    def forward(self, x, cond):
        rgb_out = self.rgb_cond_inj(x, cond)
        fft_out = self.fft_cond_inj(x, cond)

        fuse_out = torch.cat([rgb_out, fft_out], dim=1)

        out = self.to_out(fuse_out)
        return out


class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        encoder=True,
        # with_freq=None,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.encoder = encoder

        self.with_cond = exists(cond_dim)
        self.res_block = ResnetBlock(
            dim_out if exists(cond_dim) else dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups, n_head=8)
        if self.with_cond:
            if encoder:
                self.cond_inj = CondInjection(
                    dim, cond_dim, hidden_dim=dim_out, groups=norm_groups
                )
            else:
                self.cond_inj = FastAttnCondInjection(
                    dim,
                    cond_dim,
                    dim,
                    dim_out,
                    groups=norm_groups,
                    nheads=8,
                    drop_path_prob=0.2,
                )

    def forward(self, x, time_emb, cond=None):
        # condition injection
        if self.with_cond:
            x = self.cond_inj(
                x, F.interpolate(cond, size=x.shape[-2:], mode="bilinear")
            )

        # if self.with_freq:
        #     x = self.cond_inj_freq(
        #         x, F.interpolate(cond, size=x.shape[-2:], mode="bilinear")
        #     )

        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    # context to test runtime
    import contextlib
    import time
    
    @contextlib.contextmanager
    def time_it(t=10):
        t1 = time.time()
        yield
        t2 = time.time()
        print('total time: {}, ave time: {:.3f}s'.format((t2-t1), (t2 - t1)/t))
        
    device = 'cuda:1'

    net = UNetSR3(
        in_channel=31,
        channel_mults=(1, 2, 2, 2),
        out_channel=8,
        lms_channel=8,
        pan_channel=1,
        image_size=64,
        self_condition=False,
        inner_channel=32,
        norm_groups=1,
        attn_res=(8,),
        dropout=0.2,
    ).to(device)
    x = torch.randn(1, 31, 512, 512).to(device)
    cond = torch.randn(1, 31 + 1 + 31 + 3, 512, 512).to(device)  # [lms, pan, lms_main, h, v, d]
    t = torch.LongTensor([1]).to(device)
    
    with torch.no_grad():
        y = net(x, t, cond)
        tt = 25
        with time_it(tt):
            for _ in range(tt):
                y = net(x, t, cond)
                

        
    
    
    # print(y.shape)

    # print(flop_count_table(FlopCountAnalysis(net, (x, t, cond))))
    
