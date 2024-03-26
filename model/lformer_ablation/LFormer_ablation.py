import einops
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


from model.base_model import register_model, BaseModel


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


# TODO:
# 1. 不加resblock换v
# 2. 加上resblok不换v

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
            nn.Conv2d(inner_dim, inner_dim * 2, 1, bias=True),
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
        self.pan_feature_update = nn.Sequential(
                nn.Conv2d(pan_dim, inner_dim, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(inner_dim, inner_dim, 3, 1, 1))  # modified by hjm
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, lms, pan):
        *_, h, w = lms.shape

        lms = self.ms_pre_norm(lms)
        pan = self.pan_pre_norm(pan)
        # print(pan.shape)
        if self.first_layer:
            k, v = self.kv(pan).chunk(2, dim=1)
        else:
            pan_fea = self.pan_feature_update(pan)
            k, v = self.kv(pan_fea).chunk(2, dim=1)
        # k, v = self.kv(pan).chunk(2, dim=1)
        q = self.q(lms)  # q: pan
        # k, v = self.kv(pan_fea).chunk(2, dim=1)  # kv: lms
        q, k, v = map(lambda x: self.rearrange(x), (q, k, v))
        # q, k, v = map(lambda x: F.normalize(x, dim=-1), (q, k, v))

        attn = torch.einsum("b h d n, b h e n -> b h d e", q, k)  # lms x pan
        attn = self.attn_drop(attn)

        attn = attn.softmax(-1)

        out = torch.einsum("b h d e, b h e m -> b h d m", attn, v)
        out = rearrange(
            out, "b nhead c (h w) -> b (nhead c) h w", nhead=self.nheads, h=h, w=w
        )

        return attn, out, pan_fea


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
        if not first_stage:
            self.update_lms = nn.Sequential(
                Resblock(dim),
                Resblock(dim),
            )
            self.kv_conv = nn.Conv2d(dim, dim * 2, 3, 1, 1)
        self.fuse_conv = nn.Conv2d(dim + hp_dim, dim, 3, 1, 1)
        # self.reflash_attn = ReflashAttn(nhead=nhead)

        self.nhead = nhead
        self.first_stage = first_stage

    # def forward(self, reuse_attn, lms, hp_in):
    def forward(self, pan, lms, hp_in):
        *_, h, w = lms.shape

        if not self.first_stage:
            lms = self.update_lms(lms)
            # lms = rearrange(lms, "b (nhead c) h w -> b nhead c (h w)", nhead=self.nhead)
            # reuse_attn = self.reflash_attn(reuse_attn)

            q = einops.rearrange(lms, "b (nhead c) h w -> b nhead c (h w)", nhead=8)
            k, v = self.kv_conv(pan).chunk(2, dim=1)
            # print(k.shape)
            k, v = map(lambda x: einops.rearrange(x, "b (nhead c) h w -> b nhead c (h w)", nhead=8), (k, v))

            attn = torch.einsum("b h d n, b h e n -> b h d e", q, k).softmax(-1)  # lms x pan

            refined_lms = torch.einsum("b h d e, b h e m -> b h d m", attn, v)  # (lms x pan) x lms
            refined_lms = rearrange(
                refined_lms,
                "b nhead c (h w) -> b (nhead c) h w",
                nhead=self.nhead,
                h=h,
                w=w,
            )
            # print('----------------not first layer attn--------------------')
        else:
            refined_lms = lms
        # print(refined_lms.shape)
        refined_lms = self.res_block(refined_lms)

        reverse_out = torch.cat([refined_lms, hp_in], dim=1)
        out = self.fuse_conv(reverse_out)

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


@register_model("lformer_ablation")
class AttnFuseMain(BaseModel):
    def __init__(self, pan_dim, lms_dim, attn_dim, hp_dim, n_stage=3) -> None:
        super().__init__()
        self.n_stage = n_stage

        self.attn = AttnFuse(pan_dim, lms_dim, attn_dim, first_layer=False)
        self.pre_hp = PreHp(pan_dim, lms_dim, hp_dim)

        self.refined_blocks = nn.ModuleList([])
        self.hp_branch = nn.ModuleList([])
        # self.pan_update_conv = nn.Sequential(
        #     nn.Conv2d(pan_dim,attn_dim,3,1,1),
        #     nn.ReLU(),
        #     nn.Conv2d(attn_dim, attn_dim, 3, 1, 1)) # pan update

        for i in range(n_stage):
            b = MSReversibleRefine(attn_dim, hp_dim, first_stage=(i == 0))
            self.refined_blocks.append(b)

        for i in range(n_stage + 1):
            b = HpBranch(attn_dim, hp_dim)
            self.hp_branch.append(b)

        # self.final_conv = nn.Conv2d(hp_dim, lms_dim, 1)
        self.final_conv = nn.Sequential(
            Resblock(hp_dim * 2),
            Resblock(hp_dim * 2),
            nn.Conv2d(hp_dim * 2, lms_dim, 1)
        )

    def _forward_implem(self, lms, pan):
        reused_attn, refined_lms, pan_fea = self.attn(lms, pan)
        # reused_attn, refined_lms = self.attn(lms, pan)
        # pan_fea = self.pan_update_conv(pan)
        pre_hp = self.pre_hp(lms, pan)

        for i in range(self.n_stage):
            reversed_out = self.hp_branch[i](refined_lms, pre_hp)
            refined_lms = self.refined_blocks[i](pan_fea, refined_lms, reversed_out)

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
        pred = self._forward_implem(lms, pan)
        out = pred + lms

        return out.clip(0, 1)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from functools import partial

    # w/o smrb ahead qkv
    # q is pan k,v are lms: SAM 3.37
    # q is lms k,v are pan

    def _only_for_flops_count_forward(self, *args, **kwargs):
        return self._forward_implem(*args, **kwargs)

    lms = torch.randn(1, 8, 64, 64)  # .cuda(1)
    pan = torch.randn(1, 1, 64, 64)  # .cuda(1)
    gt = torch.randn(1, 8, 64, 64)  # .cuda(1)

    loss = nn.MSELoss()

    net = AttnFuseMain(pan_dim=1, lms_dim=8, attn_dim=64, hp_dim=64, n_stage=5)  # .cuda(1)

    pred = net._forward_implem(lms, pan)
    # l = loss(pred, gt)
    # l.backward()
    #
    # print(l)

    net.forward = partial(_only_for_flops_count_forward, net)

    print(flop_count_table(FlopCountAnalysis(net, (lms, pan))))
    print(net(lms, pan).shape)
