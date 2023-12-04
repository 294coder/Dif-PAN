from typing import List
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.diffusion_ddpm_google import GaussianDiffusion
from solver.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper
from unet_model_google import Upsample
from utils.misc import (
    concat_dif_timesteps_tensor,
    exist,
    img_batch2one_img,
    list_tensor2_list_list,
    model_froze,
    norm_data_range,
    unnorm_data_range,
)
from utils.model_hook import get_inner_feature_hook, hook_model


def conv3x3(dim, dim2):
    return nn.Sequential(
        nn.Conv2d(dim, dim, 3, 1, padding=1, groups=dim),
        nn.BatchNorm2d(dim),
        nn.GELU(),
        nn.Conv2d(dim, dim2, 1, 1),
    )


class UP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x = self.conv(self.up(x))
        return x


class Down(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, inplane, outplane, hidden_dim, down_up_sample=None):
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.conv1 = conv3x3(inplane, inplane)
        self.conv2 = conv3x3(inplane, hidden_dim)

        self.adaptive_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inplane, inplane // 2, bias=False),
            nn.ReLU(),
            nn.Linear(inplane // 2, outplane, bias=False),
            nn.Sigmoid(),
        )
        self.down_up_sample = (
            nn.Sequential(
                down_up_sample(hidden_dim), nn.Conv2d(hidden_dim, outplane, 1, 1)
            )
            if exist(down_up_sample)
            else conv3x3(hidden_dim, outplane)
        )

    def forward(self, x):
        b, c = x.shape[:2]
        res = x
        x = self.conv1(x)
        x = self.conv2(x)

        adap_x = self.adaptive_pooling(x).view(b, c)
        x2 = self.fc(adap_x).view(b, c, 1, 1)
        x = x * x2.expand_as(x)

        if self.inplane == self.outplane:
            x = x + res

        x = self.down_up_sample(x)

        return x


def normalized_sum(x):
    out = 0.0
    for i in range(len(x)):
        xi = x[i]
        out = out + F.normalize(xi, p=2, dim=1)
    return out


class MultiScaleModel(nn.Module):
    def __init__(
        self,
        unet: nn.Module = None,
        inplane: int = 17,
        dims: List[int] = [32, 64, 128, 64, 64, 8],
        hook_gen_model=True,
        pre_cal_fm=False,
        down_sample_place=[0, 1],
        up_sample_place=[2, 3],
    ) -> None:
        super().__init__()

        assert (not pre_cal_fm and exist(unet)) or (
            pre_cal_fm and not exist(unet)
        ), f"@pre_cal_fm and @unet should be [False, nn.Module] or [True, None], but get [{pre_cal_fm}, {unet}]"
        if not pre_cal_fm and exist(unet):
            # diffusion unet
            self.unet = unet
            self.unet.eval()
            model_froze(self.unet)
            if hook_gen_model:
                self.diffusion_fms = []
                hook_model(
                    self.unet, get_inner_feature_hook, Upsample, self.diffusion_fms
                )

        # plain decoder
        assert len(down_sample_place) == len(
            up_sample_place
        ), "downsample scale should equal to upsample scale"
        self.down_sample_place = down_sample_place
        self.up_sample_place = up_sample_place
        self.conv = nn.Conv2d(inplane, dims[0], 1, 1)
        _ml = []
        for i in range(len(dims) - 1):
            d = dims[i]
            d2 = dims[i + 1]

            if i in down_sample_place:
                down_up_sample = Down
            elif i in up_sample_place:
                down_up_sample = UP
            else:
                down_up_sample = None
            _ml.append(ResBlock(d, d, d, down_up_sample=down_up_sample))
            if i != len(dims) - 1:
                _ml.append(nn.Conv2d(d, d2, 1, 1))

        self.plain_decoder = nn.ModuleList(_ml)

    def forward(
        self,
        lms,
        pan,
        schedule,
        n,
        pre_cal_fm: List[List[torch.Tensor]] = None,
        **solver_kwargs,
    ):
        b = pan.shape[0]
        cond = torch.cat([lms, pan], dim=1)
        # get feature map
        if not exist(pre_cal_fm):
            with torch.no_grad():
                self.diffusion_fms.clear()
                model_fn = model_wrapper(
                    self.unet,
                    schedule,
                    guidance_type="classifier-free",
                    guidance_scale=1.0,
                    condition=norm_data_range(cond),  # norm it to [-1, 1]
                    model_kwargs={"interm_fm": True},
                )
                solver = DPM_Solver(
                    model_fn, schedule, correcting_x0_fn=lambda x, t: x.clamp(-1.0, 1.0)
                )
                sampled_sr = solver.sample(torch.randn_like(lms), **solver_kwargs)
                # sr = img_batch2one_img(sampled_sr).numpy()
                # sr -= sr.min()
                # sr /= sr.max()
                # plt.imshow(sr)
                # fig = plt.gcf()
                fms = list_tensor2_list_list(self.diffusion_fms, n=n, mode="size")
        else:
            sampled_sr = pre_cal_fm[0]
            fms = pre_cal_fm[1:]

        # sr = self.diffusion_fms[-1][0, 0].detach().cpu().numpy()
        # sr -= sr.min()
        # sr /= sr.max()
        # plt.imshow(sr)
        # fig = plt.gcf()
        
        x = torch.cat([cond, unnorm_data_range(sampled_sr.to(cond.device))], dim=1)
        # x = cond
        x = self.conv(x)
        first_fm_fuse_place = self.up_sample_place[0]
        for i, d in enumerate(self.plain_decoder):
            if i >= first_fm_fuse_place * 2 and i % 2 == 0:
                index = i // 2 - len(self.down_sample_place)
                fm = fms[index]  # n feature maps in timestep t_i
                x = x + normalized_sum(fm)

            x = d(x)
        return x + lms


if __name__ == "__main__":
    from unet_model_google import UNet

    image_n_channel = 8
    image_size = 64
    device = "cuda:0"
    unet = UNet(
        in_channel=image_n_channel * 2 + 1,
        out_channel=image_n_channel,
        image_size=image_size,
        res_blocks=2,
        attn_res=(8, 16),
        channel_mults=(1, 2, 2, 4),
    ).to(device)
    schedule = dict(
        schedule="cosine",
        n_timestep=3000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    )
    diffusion = GaussianDiffusion(
        unet,
        image_size,
        channels=image_n_channel,
        loss_type="l2",
        schedule_opt=schedule,
        device=device,
        conditional=True,
        clamp_range=(-1.0, 1.0),
    )
    schedule = NoiseScheduleVP(betas=diffusion.betas)
    pan_net = MultiScaleModel(unet, 9).to(device)
    x = pan_net(
        torch.randn(1, 8, 64, 64).to(device),
        torch.randn(1, 1, 64, 64).to(device),
        schedule,
        n=3,
        steps=100,
        order=2,
    )

    print(x.shape)
