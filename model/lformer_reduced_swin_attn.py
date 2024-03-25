import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

import sys
sys.path.append('./')

from model.base_model import register_model, BaseModel
from utils import get_local

# get_local.activate()


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
    def __init__(self, nhead=8, ksize=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                nhead, nhead, (1, ksize), stride=1, padding=(0, ksize // 2), bias=False
            ),
            nn.Conv2d(nhead, nhead, 1, bias=True),
            nn.ReLU(),
        )
        self.body[0].weight.data.fill_(1.0 / (ksize * ksize))
        self.body[1].weight.data.fill_(1.0)
        self.body[1].bias.data.fill_(0.0)

    def forward(self, attn):
        return self.body(attn).softmax(-1)
    
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # B, C, H, W = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = rearrange(x, "b c (h w1) (w w2) -> (b w1 w2) c h w", h=window_size, w=window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = rearrange(windows, '(b w1 w2) c h w -> b c (h w1) (w w2)', 
                  b=B, w1=H//window_size, w2=W//window_size)
    return x


# TODO:
# 1. 不加resblock换v
# 2. 加上resblok不换v


class FirstAttn(nn.Module):
    def __init__(
        self, 
        pan_dim,
        lms_dim,
        inner_dim,
        nheads=8, 
        attn_drop=0.2,
        first_layer=False,
        *,
        attn_type='W',
        window_size=16,
        r_op:callable=None
    ) -> None:
        super().__init__()
        self.nheads = nheads
        self.first_layer = first_layer
        
        assert attn_type in ['R', 'W', 'I'], 'R is reduced attn, W is window attn, I is tranditional attn'
        if attn_type == 'W': 
            assert window_size is not None, 'window_size should be given'
        elif attn_type == 'R':
            self.r_op = r_op
        else:
            self.r_op = nn.Identity()
        self.attn_type = attn_type
        self.ws = window_size

        self.rearrange = Rearrange("b (nhead c) h w -> b nhead c (h w)", nhead=nheads)
        self.q = nn.Sequential(
            nn.Conv2d(pan_dim, inner_dim, 1, bias=True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, groups=inner_dim),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(lms_dim, inner_dim * 2, 1, bias=True),
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
        
    def R_attn_forwawrd(self, lms, pan):
        *_, h, w = lms.shape
        
        lms = self.ms_pre_norm(lms)
        pan = self.pan_pre_norm(pan)
        
        q = self.q(pan)  # q: pan
        k, v = self.kv(lms).chunk(2, dim=1)  # kv: lms
        
        k, v = self.r_op(k), self.r_op(v)
        
        q, k, v = map(lambda x: self.rearrange(x), (q, k, v))
        
        attn = torch.einsum("b h d m, b h d n -> b h m n", q, k)  # lms x pan
        attn = self.attn_drop(attn)

        attn = attn.softmax(-1)

        out = torch.einsum("b h m n, b h d n -> b h d m", attn, v)
        out = rearrange(
            out, "b nhead d (h w) -> b (nhead d) h w", nhead=self.nheads, h=h, w=w
        )
        
        return attn, out

    # @get_local("attn", "out")
    def W_attn_forward(self, lms, pan):
        *_, h, w = lms.shape

        lms = self.ms_pre_norm(lms)
        pan = self.pan_pre_norm(pan)
        
        q = self.q(pan)  # q: pan
        k, v = self.kv(lms).chunk(2, dim=1)  # kv: lms

        q, k, v = map(lambda x: self.rearrange(window_partition(x, self.ws)), (q, k, v))
        # q, k, v = map(lambda x: F.normalize(x, dim=-1), (q, k, v))
        
        k = F.normalize(k, dim=-1)

        attn = torch.einsum("b h d m, b h d n -> b h m n", q, k)  # lms x pan
        attn = self.attn_drop(attn)

        attn = attn.softmax(-1)

        out = torch.einsum("b h m n, b h d n -> b h d m", attn, v)
        out = rearrange(
            out, "b nhead d (h w) -> b (nhead d) h w", nhead=self.nheads, h=self.ws, w=self.ws
        )
        out = window_reverse(out, self.ws, h, w)
        
        return attn, out
    
    def forward(self, lms, pan):
        if self.attn_type == 'W':
            return self.W_attn_forward(lms, pan)
        else:
            return self.R_attn_forwawrd(lms, pan)


class MSReversibleRefine(nn.Module):
    def __init__(self,
                 dim,
                 hp_dim,
                 nhead=8,
                 first_stage=False,
                 window_size=16,
                 attn_type='W',
                 r_op:callable=None) -> None:
        super().__init__()
        if attn_type == 'R':
            if r_op is None:
                raise ValueError('r_op must be set to a callable function')
        
        self.attn_type = attn_type
        self.ws = window_size
        self.r_op = r_op
        self.res_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1),
                # nn.BatchNorm2d(dim),
            )
        )
        self.fuse_conv = nn.Conv2d(dim + hp_dim, dim, 3, 1, 1)
        if not first_stage:
            self.reflash_attn = ReflashAttn(nhead=nhead)

        self.nhead = nhead
        self.first_stage = first_stage

    # @get_local("reflashed_attn", "reflashed_out")
    def forward(self, reused_attn, refined_lms, hp_in):
        *_, h, w = refined_lms.shape

        if not self.first_stage:
            reused_attn = self.reflash_attn(reused_attn)
            if self.attn_type == 'W':
                refined_lms = window_partition(refined_lms, self.ws)
                refined_lms = rearrange(refined_lms, "b (nhead c) h w -> b nhead c (h w)", nhead=self.nhead)

                refined_lms = torch.einsum(
                    "b h m n, b h d m -> b h d n", reused_attn, refined_lms
                )  # (lms x pan) x lms
                refined_lms = rearrange(
                    refined_lms,
                    "b nhead c (h w) -> b (nhead c) h w",
                    nhead=self.nhead,
                    h=self.ws,
                    w=self.ws,
                )
                refined_lms = window_reverse(refined_lms, self.ws, h, w)
            else:
                refined_lms = self.r_op(refined_lms)
                refined_lms = rearrange(refined_lms, "b (nhead c) h w -> b nhead c (h w)", nhead=self.nhead)

                refined_lms = torch.einsum(
                    "b h m n, b h d n -> b h d m", reused_attn, refined_lms
                )  # (lms x pan) x lms
                refined_lms = rearrange(
                    refined_lms,
                    "b nhead c (h w) -> b (nhead c) h w",
                    nhead=self.nhead,
                    h=h,
                    w=w,
                )
            
            
        # reflashed_attn = reuse_attn
        reflashed_out = refined_lms
        refined_lms = self.res_block(refined_lms)
        reverse_out = torch.cat([refined_lms, hp_in], dim=1)
        out = self.fuse_conv(reverse_out)
        out = reflashed_out * out

        return reused_attn, out


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
                nn.Conv2d(hp_dim, hp_dim, 1),
                # nn.BatchNorm2d(hp_dim),
            )
        )

    def forward(self, refined_lms, hp_in):
        attn_hp = self.attn_hp_conv(refined_lms)
        x = torch.cat([attn_hp, hp_in], dim=1)
        hp_out = self.res_block(self.to_hp_dim(x))
        return hp_out


@register_model("lformer_R")
# buggy: training unstable
class AttnFuseMain(BaseModel):
    def __init__(
        self,
        pan_dim,
        lms_dim,
        attn_dim,
        hp_dim,
        n_stage=3,
        patch_merge=True,
        crop_batch_size=1,
        patch_size_list=None,
        scale=4,
        attn_type='R',
        r_op=None,
    ) -> None:
        super().__init__()
        self.n_stage = n_stage

        if attn_type == 'R' and r_op is None:
            # TODO: input img_size?
            r_op = nn.AdaptiveAvgPool2d((16, 16))
            
        self.attn = FirstAttn(pan_dim, lms_dim, attn_dim, first_layer=False, attn_type=attn_type, r_op=r_op)
        self.pre_hp = PreHp(pan_dim, lms_dim, hp_dim)

        self.refined_blocks = nn.ModuleList([])
        self.hp_branch = nn.ModuleList([])
        for i in range(n_stage):
            b = MSReversibleRefine(attn_dim, hp_dim, first_stage=(i == 0), attn_type=attn_type, r_op=r_op)
            self.refined_blocks.append(b)

        for i in range(n_stage + 1):
            b = HpBranch(attn_dim, hp_dim)
            self.hp_branch.append(b)

        # self.final_conv = nn.Conv2d(hp_dim, lms_dim, 1)
        self.final_conv = nn.Sequential(
            Resblock(hp_dim + attn_dim),
            Resblock(hp_dim + attn_dim),
            nn.Conv2d(hp_dim + attn_dim, lms_dim, 1),
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
        reused_attn, refined_lms = self.attn(lms, pan)
        pre_hp = self.pre_hp(lms, pan)
        # print(reused_attn.shape)
        for i in range(self.n_stage):
            reversed_out = self.hp_branch[i](refined_lms, pre_hp)
            # TODO: update reused_attn or not
            reused_attn, refined_lms = self.refined_blocks[i](
                reused_attn, refined_lms, reversed_out
            )

        out = self.hp_branch[-1](refined_lms, pre_hp)
        out = torch.cat([out, refined_lms], dim=1)

        out = self.final_conv(out)

        return out

    def train_step(self, ms, lms, pan, gt, criterion):
        pred = self._forward_implem(lms, pan)

        out = pred + lms
        loss = criterion(out, gt)
        return out.clip(0, 1), loss

    def val_step(self, ms, lms, pan, patch_merge=True):
        if self.patch_merge and patch_merge:
            pred = self._patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            pred = self._forward_implem(lms, pan)
        out = pred + lms

        return out.clip(0, 1)

    def patch_merge_step(self, ms, lms, pan):
        return self._forward_implem(lms, pan)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from functools import partial

    # w/o smrb ahead qkv
    # q is pan k,v are lms: SAM 3.37
    # q is lms k,v are pan
    
    torch.cuda.set_device(0)

    def _only_for_flops_count_forward(self, *args, **kwargs):
        return self._forward_implem(*args, **kwargs)

    ms = torch.randn(1, 8, 16, 16).cuda()
    lms = torch.randn(1, 8, 64, 64).cuda()
    pan = torch.randn(1, 1, 64, 64).cuda()
    
    net = AttnFuseMain(
        pan_dim=1,
        lms_dim=8,
        attn_dim=64,
        hp_dim=64,
        n_stage=5,
        patch_merge=False,
        patch_size_list=[16, 64, 64],
        scale=4,
        crop_batch_size=32,
        attn_type='R',
        r_op=nn.AdaptiveAvgPool2d((16, 16))
    ).cuda()
    net.forward = partial(_only_for_flops_count_forward, net)

    # sr = net.val_step(ms, lms, pan)
    # sr = net._forward_implem(lms, pan)
    # loss = F.mse_loss(sr, torch.randn(1, 8, 64, 64).cuda()).backward()
    # print(sr.shape)
    
    # for n, m in net.named_parameters():
    #     if m.grad is None:
    #         print(f'{n} has no grad')

    print(flop_count_table(FlopCountAnalysis(net, (lms, pan))))
    # print(net(lms, pan).shape)

    ## dataset: num_channel HSI/PAN
    # Pavia: 102/1
    # botswana: 145/1
    # chikusei: 128/3

    # num_p = 0
    # for p in net.parameters():
    #     if p.requires_grad:
    #         num_p += p.numel()

    # print(num_p/1e6)

    # cache = get_local.cache
    # print(cache)