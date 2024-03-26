"""
Abalation on Flatten Transformer Block

replace the LFormer Blocks (include the first SA) with Flatten Transformer Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from beartype import beartype
from typing import Tuple, List, Dict, Union, Optional, Any

# from utils.visualize import get_local
# get_local.activate()

from model.base_model import register_model, BaseModel

# modified into cross attention
class FocusedLinearCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim_lms, dim_pan, inner_dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = inner_dim
        # self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads

        self.focusing_factor = focusing_factor
        self.q = nn.Linear(dim_lms, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim_pan, inner_dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, inner_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, inner_dim)))
        # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))
        # print('Linear Attention window{} f{} kernel{}'.
        #       format(window_size, focusing_factor, kernel_size))

    def forward(self, lms, pan, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        
        h, w = pan.shape[-2:]
        lms = rearrange(lms, 'b c h w -> b (h w) c')
        pan = rearrange(pan, 'b c h w -> b (h w) c')
        
        q = self.q(lms)
        kv = self.kv(pan)
        k, v = kv.chunk(2, dim=-1)
        # k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        # kernel_function = nn.ReLU()
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        scale = F.softplus(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
    
class FocusedLinearSelfAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, inner_dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = inner_dim
        # self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(inner_dim, inner_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, inner_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, inner_dim)))
        # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))
        # print('Linear Attention window{} f{} kernel{}'.
        #       format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        # kernel_function = nn.ReLU()
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        scale = F.softplus(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)  


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

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Resblock(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class LayerNorm2d(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class ReflashValue(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim * 2, 1, bias=True),
        )

    def forward(self, v, v2):
        scale, shift = self.body(v2).chunk(2, dim=1)
        return v * (1 + scale) + shift


class ReflashAttn(nn.Module):
    def __init__(self, nhead=8, ksize=5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nhead, nhead, (1, ksize), stride=1, padding=(0, ksize // 2), bias=False),
            nn.Conv2d(nhead, nhead, 1, bias=True),
            nn.ReLU()
        )
        self.body[0].weight.data.fill_(1.0 / (ksize * ksize))
        self.body[1].weight.data.fill_(1.0)
        self.body[1].bias.data.fill_(0.0)

    def forward(self, attn):
        return self.body(attn).softmax(-1)
    
    
class WarpAttnModules(nn.Module):
    def __init__(self, module_name='F', *args, **kwargs):
        super().__init__()
        self.mn = module_name
        
        if module_name == 'F':
            self.attn = FocusedLinearCrossAttention(*args, **kwargs)
            
        elif module_name == 'L':
            self.attn = AttnFuse(*args, **kwargs)
            
        elif module_name == 'M':
            self.attn = MSReversibleRefineFlattenAttn(*args, **kwargs)
            
        else:
            raise NotImplementedError(f'Not support module name: {module_name}')
        
    def forward(self, reused_attn=None, lms=None, pan=None):
        if self.mn in ['F', 'L']:
            return self.attn(lms, pan)
        else:
            return self.attn(reused_attn, lms, pan)

class AttnFuse(nn.Module):
    def __init__(
            self, pan_dim, lms_dim, inner_dim, nheads=8, attn_drop=0.2, first_layer=False
    ) -> None:
        super().__init__()
        self.nheads = nheads
        self.first_layer = first_layer

        self.rearrange = Rearrange("b (nhead c) h w -> b nhead c (h w)", nhead=nheads)
        self.q = nn.Sequential(
            nn.Conv2d(lms_dim, inner_dim, 1, bias=True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, groups=inner_dim),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(pan_dim, inner_dim * 2, 1, bias=True),
            nn.Conv2d(inner_dim * 2, inner_dim * 2, 3, 1, 1, groups=inner_dim * 2),
        )
        if first_layer:
            self.q.append(
                nn.Sequential(
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                )
            )
            self.kv.append(
                nn.Sequential(
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                )
            )

        self.ms_pre_norm = LayerNorm2d(lms_dim)
        self.pan_pre_norm = LayerNorm2d(pan_dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, lms, pan):
        *_, h, w = lms.shape

        lms = self.ms_pre_norm(lms)
        pan = self.pan_pre_norm(pan)

        q = self.q(lms)  # q: pan
        k, v = self.kv(pan).chunk(2, dim=1)  # kv: lms

        q, k, v = map(lambda x: self.rearrange(x), (q, k, v))
        # q, k, v = map(lambda x: F.normalize(x, dim=-1), (q, k, v))

        attn = torch.einsum("b h d n, b h e n -> b h d e", q, k)  # lms x pan
        attn = self.attn_drop(attn)

        attn = attn.softmax(-1)

        out = torch.einsum("b h d e, b h e m -> b h d m", attn, v)
        out = rearrange(
            out, "b nhead c (h w) -> b (nhead c) h w", nhead=self.nheads, h=h, w=w
        )

        return attn, out
    
    
class MSReversibleRefineFlattenAttn(nn.Module):
    def __init__(self, dim, hp_dim, nhead=8, first_stage=False) -> None:
        super().__init__()
        self.res_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1)
                # nn.BatchNorm2d(dim),
            )
        )
        self.fuse_conv = nn.Conv2d(dim + hp_dim, dim, 3, 1, 1)
        self.flattn_attn = FocusedLinearSelfAttention(inner_dim=dim+hp_dim, num_heads=nhead)

        self.nhead = nhead
        self.first_stage = first_stage

    def forward(self, reused_attn, refined_lms, hp_in):
        # *_, h, w = lms.shape
        
        refined_lms = self.res_block(refined_lms)
        refined_lms = torch.cat([refined_lms, hp_in], dim=1)

        if not self.first_stage:
            # lms = rearrange(lms, "b (nhead c) h w -> b nhead c (h w)", nhead=self.nhead)
            # # print(reuse_attn.shape)
            # reuse_attn = self.(reuse_attn)
            # refined_lms = torch.einsum("b h d e, b h e m -> b h d m", reuse_attn, lms)  # (lms x pan) x lms
            # refined_lms = rearrange(
            #     refined_lms,
            #     "b nhead c (h w) -> b (nhead c) h w",
            #     nhead=self.nhead,
            #     h=h,
            #     w=w,
            # )
            reverse_out = self.flattn_attn(refined_lms)
        else:
            reverse_out = refined_lms

        out = self.fuse_conv(reverse_out)
        
        # print(refined_lms.shape, out.shape)
        # print('-'*30)

        return out


class MSReversibleRefine(nn.Module):
    def __init__(self, dim, hp_dim, nhead=8, first_stage=False) -> None:
        super().__init__()
        self.res_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1)
                # nn.BatchNorm2d(dim),
            )
        )
        self.fuse_conv = nn.Conv2d(dim + hp_dim, dim, 3, 1, 1)
        self.reflash_attn = ReflashAttn(nhead=nhead)

        self.nhead = nhead
        self.first_stage = first_stage

    def forward(self, reuse_attn, lms, hp_in):
        *_, h, w = lms.shape

        if not self.first_stage:
            lms = rearrange(lms, "b (nhead c) h w -> b nhead c (h w)", nhead=self.nhead)
            # print(reuse_attn.shape)
            reuse_attn = self.reflash_attn(reuse_attn)
            refined_lms = torch.einsum("b h d e, b h e m -> b h d m", reuse_attn, lms)  # (lms x pan) x lms
            refined_lms = rearrange(
                refined_lms,
                "b nhead c (h w) -> b (nhead c) h w",
                nhead=self.nhead,
                h=h,
                w=w,
            )
        else:
            refined_lms = lms

        refined_lms = self.res_block(refined_lms)

        reverse_out = torch.cat([refined_lms, hp_in], dim=1)
        out = self.fuse_conv(reverse_out)
        
        # print(refined_lms.shape, out.shape)
        # print('-'*30)

        return out


class PreHp(nn.Module):
    def __init__(self, pan_dim, lms_dim, hp_dim) -> None:
        super().__init__()
        self.hp_pan = nn.Sequential(
            nn.Conv2d(pan_dim, pan_dim, 3, 1, 1, groups=pan_dim),
            nn.Conv2d(pan_dim, hp_dim, 1),
        )
        self.hp_lms = nn.Sequential(
            nn.Conv2d(lms_dim, lms_dim, 3, 1, 1, groups=lms_dim),
            nn.Conv2d(lms_dim, hp_dim, 1),
        )
        # laplacian init
        self.hp_pan[0].weight.data = torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32
        ).repeat(pan_dim, 1, 1, 1)
        self.hp_pan[0].bias.data = torch.tensor([0.0] * pan_dim, dtype=torch.float32)
        self.hp_lms[0].weight.data = torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32
        ).repeat(lms_dim, 1, 1, 1)
        self.hp_lms[0].bias.data = torch.tensor([0.0] * lms_dim, dtype=torch.float32)

        self.hp_conv = nn.Conv2d(hp_dim * 2, hp_dim, 1)

        self.msrb = nn.Sequential(
            Resblock(hp_dim), Resblock(hp_dim), Resblock(hp_dim), Resblock(hp_dim)
        )

    def forward(self, lms, pan):
        hp_pan = self.hp_pan(pan)
        hp_lms = self.hp_lms(lms)
        x = torch.cat([hp_pan, hp_lms], dim=1)
        hp_in = self.msrb(self.hp_conv(x))
        return hp_in


class HpBranch(nn.Module):
    def __init__(self, attn_dim, hp_dim) -> None:
        super().__init__()
        self.attn_hp_conv = nn.Conv2d(attn_dim, hp_dim, 1)
        self.to_hp_dim = nn.Conv2d(hp_dim * 2, hp_dim, 1)

        self.res_block = Residual(
            nn.Sequential(
                nn.Conv2d(hp_dim, hp_dim, 3, 1, 1, groups=hp_dim),
                nn.ReLU(),
                nn.Conv2d(hp_dim, hp_dim, 1)
                # nn.BatchNorm2d(hp_dim),
            )
        )

    def forward(self, refined_lms, hp_in):
        attn_hp = self.attn_hp_conv(refined_lms)
        x = torch.cat([attn_hp, hp_in], dim=1)
        hp_out = self.res_block(self.to_hp_dim(x))
        return hp_out


@register_model("lformer_cvpr_flatten_transformer")
class AttnFuseMain(BaseModel):
    def __init__(self, pan_dim, lms_dim, attn_dim, hp_dim, n_stage=3,
                 patch_merge=True, crop_batch_size=1, patch_size_list=None,
                 scale=4) -> None:
        super().__init__()
        self.n_stage = n_stage

        # self.attn = AttnFuse(pan_dim, lms_dim, attn_dim, first_layer=False)
        self.attn = WarpAttnModules('F', lms_dim, pan_dim, attn_dim, num_heads=8)
        self.pre_hp = PreHp(pan_dim, lms_dim, hp_dim)

        self.refined_blocks = nn.ModuleList([])
        self.hp_branch = nn.ModuleList([])
        for i in range(n_stage):
            # b = MSReversibleRefine(attn_dim, hp_dim, first_stage=(i == 0))
            b = WarpAttnModules('M', attn_dim, hp_dim, first_stage=(i == 0))
            self.refined_blocks.append(b)

        for i in range(n_stage + 1):
            b = HpBranch(attn_dim, hp_dim)
            self.hp_branch.append(b)

        # self.final_conv = nn.Conv2d(hp_dim, lms_dim, 1)
        self.final_conv = nn.Sequential(
            Resblock(hp_dim + attn_dim),
            Resblock(hp_dim + attn_dim),
            nn.Conv2d(hp_dim + attn_dim, lms_dim, 1)
        )
        
        self.patch_merge = patch_merge
        if patch_merge:
            from model.base_model import PatchMergeModule
            self._patch_merge_model = PatchMergeModule(
                # net=self,
                patch_merge_step=self.patch_merge_step,
                crop_batch_size=crop_batch_size,
                patch_size_list=patch_size_list,
                scale=scale,
            )

    def _forward_implem(self, lms, pan):
        # reused_attn, refined_lms = self.attn(lms, pan)
        refined_lms = self.attn(lms=lms, pan=pan)
        pre_hp = self.pre_hp(lms, pan)
        # print(reused_attn.shape)
        for i in range(self.n_stage):
            reversed_out = self.hp_branch[i](refined_lms, pre_hp)
            refined_lms = self.refined_blocks[i](reused_attn=None, 
                                                 lms=refined_lms, 
                                                 pan=reversed_out)

        out = self.hp_branch[-1](refined_lms, pre_hp)
        out = torch.cat([out, refined_lms], dim=1)

        out = self.final_conv(out)

        return out

    def train_step(self, ms, lms, pan, gt, criterion):
        pred = self._forward_implem(lms, pan)

        out = pred + lms
        loss = criterion(out, gt)
        return out.clip(0, 1), loss

    def val_step(self, ms, lms, pan):
        if self.patch_merge:
            pred = self._patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            pred = self._forward_implem(lms, pan)
        out = pred + lms

        return out.clip(0, 1)
    
    def patch_merge_step(self, ms, lms, pan):
        return self._forward_implem(lms, pan)
    
    
# def _get_feat(key='MSReversibleRefine.forward'):
#     cache = get_local.cache
#     refined_feat = cache[key]
#     get_local.clear()
    
#     return refined_feat


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from functools import partial


    # w/o smrb ahead qkv
    # q is pan k,v are lms: SAM 3.37
    # q is lms k,v are pan

    def _only_for_flops_count_forward(self, *args, **kwargs):
        return self._forward_implem(*args, **kwargs)

    ms = torch.randn(1, 8, 16, 16).cuda(1) 
    lms = torch.randn(1, 8, 64, 64).cuda(1)
    pan = torch.randn(1, 1, 64, 64).cuda(1)

    net = AttnFuseMain(pan_dim=1, lms_dim=8, attn_dim=64, hp_dim=64, n_stage=5,
                       patch_merge=False, patch_size_list=[16,64,64], scale=4,
                       crop_batch_size=32).cuda(1)
    net.forward = partial(_only_for_flops_count_forward, net)
    
    # for _ in range(3):
    print('=='*50)
    
    # sr = net.val_step(ms, lms, pan)
    # print(sr.shape)
    
    print(flop_count_table(FlopCountAnalysis(net, (lms, pan))))
    
    # cache = get_local.cache
    # cache = _get_feat()
    # for c in cache:
    #     print(c[0].shape)
    # for i in range(len(cache)):
    #     for j in range(2):
    #         print(cache[i][j].shape)
    # pass