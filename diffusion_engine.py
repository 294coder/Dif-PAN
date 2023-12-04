from functools import partial
import time
from copy import deepcopy

import einops
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision as tv
from scipy.io import savemat
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.pan_dataset import PanDataset
from dataset.hisr import HISRDataSets
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.logger import TensorboardLogger
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import compute_iters, exist, grad_clip, model_load, path_legal_checker
from utils.optim_utils import EmaUpdater


def show_distribution(hr):
    plt.figure()
    gt = hr[:2].detach().cpu().flatten().numpy()
    sns.displot(data=gt)
    plt.show()


def norm(x):
    # x range is [0, 1]
    return x * 2 - 1


def unorm(x):
    # x range is [-1, 1]
    return (x + 1) / 2


def clamp_fn(g_sr):
    def inner(x):
        x = x + g_sr
        x = x.clamp(0, 1.0)
        return x - g_sr

    return inner


def engine_google(
    # dataset
    train_dataset_path,
    valid_dataset_path,
    dataset_name=None,
    # image settings
    image_n_channel=8,
    image_size=64,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # device setting
    device="cuda:0",
    # optimizer settings
    batch_size=128,
    lr_d=1e-4,
    show_recon=False,
    # pretrain settings
    pretrain_weight=None,
    pretrain_iterations=None,
    *,
    # just for debugging
    constrain_channel=None,
):
    """train and valid function

    Args:
        train_dataset_path (str): _description_
        valid_dataset_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 240.
        n_steps (int, optional): _description_. Defaults to 1500.
        epochs (int, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to 'cuda:0'.
        max_iterations (int, optional): _description_. Defaults to 500_000.
        lr (float, optional): _description_. Defaults to 1e-4.
        pretrain_weight (str, optional): _description_. Defaults to None.
        recon_loss (bool, optional): _description_. Defaults to False.
        show_recon (bool, optional): _description_. Defaults to False.
        constrain_channel (int, optional): _description_. Defaults to None.
    """
    from diffusion.diffusion_ddpm_pan import GaussianDiffusion
    from models.sr3_dwt import UNetSR3 as Unet

    # init logger
    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = "pandiff"
    logger = TensorboardLogger(file_logger_name="{}-{}".format(stf_time, comment))

    dataset_name = (
        train_dataset_path.strip(".h5").split("_")[-1]
        if not exist(dataset_name)
        else dataset_name
    )
    logger.print(f"dataset name: {dataset_name}")
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0, "cave": 1.0, "harvard": 1.0}
    logger.print(f"dataset norm division: {division_dict[dataset_name]}")
    rgb_channel = {
        "wv3": [4, 2, 0],
        "gf2": [0, 1, 2],
        "qb": [0, 1, 2],
        "cave": [29, 19, 9],
        "harvard": [29, 19, 9]
    }
    logger.print(f"rgb channel: {rgb_channel[dataset_name]}")
    add_n_channel = 1

    # initialize models
    torch.cuda.set_device(device)
    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=add_n_channel,
        inner_channel=32,
        norm_groups=1,
        channel_mults=(1,2,2,4),#(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)
    if pretrain_weight is not None:
        if isinstance(pretrain_weight, (list, tuple)):
            model_load(pretrain_weight[0], denoise_fn, strict=True, device=device)
        else:
            model_load(pretrain_weight, denoise_fn, strict=False, device=device)
        print("load pretrain weight from {}".format(pretrain_weight))

    # get dataset
    d_train = h5py.File(train_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=constrain_channel,
            division=division_dict[dataset_name],
            aug_prob=0,
            wavelets=True,
        )
    elif dataset_name in ["cave", "harvard"]:
        DatasetUsed = partial(HISRDataSets, normalize=False, aug_prob=0, wavelets=True)
    else:
        raise NotImplementedError("dataset {} not supported".format(dataset_name))

    ds_train = DatasetUsed(
        d_train,
    )
    ds_valid = DatasetUsed(
        d_valid,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # diffusion
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        # num_sample_steps=n_steps,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    # model, optimizer and lr scheduler
    diffusion_dp = (
        diffusion
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=1e-4)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.2
    )
    schedulers = StepsAll(scheduler_d)

    # training
    if pretrain_iterations is not None:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
    else:
        iterations = 0
    while iterations <= max_iterations:
        for i, (pan, lms, hr, wavelets) in enumerate(dl_train, 1):
            pan, lms, hr, wavelets = map(lambda x: x.cuda(), (pan, lms, hr, wavelets))
            cond, _ = einops.pack(
                [
                    lms,
                    pan,
                    F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear"),
                ],
                "b * h w",
            )

            opt_d.zero_grad()
            res = hr - lms
            diff_loss, recon_x = diffusion_dp(res, cond=cond)
            diff_loss.backward()
            recon_x = recon_x + lms

            # do a grad clip on diffusion model
            grad_clip(diffusion_dp.model.parameters(), mode="norm", value=0.003)

            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
                + f"denoise loss {diff_loss:.6f} "
            )

            # test predicted sr
            if show_recon and iterations % 1_000 == 0:
                # NOTE: only used to validate code
                recon_x = recon_x[:64]

                x = tv.utils.make_grid(recon_x, nrow=8, padding=0).cpu()
                x = x.clip(0, 1)  # for no warning
                fig, ax = plt.subplots(figsize=(x.shape[-1] // 100, x.shape[-2] // 100))
                x_show = (
                    x.permute(1, 2, 0).detach().numpy()[..., rgb_channel[dataset_name]]
                )
                ax.imshow(x_show)
                ax.set_axis_off()
                plt.tight_layout(pad=0)
                # plt.show()
                fig.savefig(
                    f"./samples/recon_x/iter_{iterations}.png",
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # do some sampling to check quality
            if iterations % 5_000 == 0:
                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()

                analysis_d = AnalysisPanAcc()
                with torch.no_grad():
                    for i, (pan, lms, hr, wavelets) in enumerate(dl_valid, 1):
                        torch.cuda.empty_cache()
                        pan, lms, hr, wavelets = map(
                            lambda x: x.cuda(), (pan, lms, hr, wavelets)
                        )

                        cond, _ = einops.pack(
                            [
                                lms,
                                pan,
                                F.interpolate(
                                    wavelets, size=lms.shape[-1], mode="bilinear"
                                ),
                            ],
                            "b * h w",
                        )

                        sr = ema_updater.ema_model(cond, mode="ddim_sample", section_counts="ddim25")
                        sr = sr + lms
                        sr = sr.clip(0, 1)

                        hr = hr.to(sr.device)
                        analysis_d(hr, sr)
                        
                        hr = tv.utils.make_grid(hr, nrow=4, padding=0).cpu()
                        x = tv.utils.make_grid(sr, nrow=4, padding=0).detach().cpu()
                        x = x.clip(0, 1)

                        s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]
                        fig, ax = plt.subplots(
                            figsize=(s.shape[-1] // 100, s.shape[-2] // 100)
                        )
                        ax.imshow(
                            s.permute(1, 2, 0)
                            .detach()
                            .numpy()[..., rgb_channel[dataset_name]]
                        )
                        ax.set_axis_off()

                        plt.tight_layout(pad=0)
                        fig.savefig(
                            f"./samples/valid_samples/iter_{iterations}.png",
                            dpi=200,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.last_acc)
                    if i != 1:
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.print_str())
                diffusion_dp.model.train()
                setattr(ema_updater.model, "image_size", 64)

                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"./weights/diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./weights/ema_diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", diff_loss.item(), iterations)


@torch.no_grad()
def test_fn(
    test_data_path,
    weight_path,
    schedule_type="cosine",
    batch_size=320,
    n_steps=1500,
    show=False,
    device="cuda:0",
    full_res=False,
    dataset_name="gf2",
    division=1023,
):
    # lazy import
    from models.sr3_dwt import UNetSR3 as Unet
    from diffusion.diffusion_ddpm_pan import GaussianDiffusion

    torch.cuda.set_device(device)

    # load model
    if dataset_name in ['harvard', 'cave']:
        image_n_channel = 31
        image_size = 512 if dataset_name == 'cave' else 1000
        pan_channel = 3
        rgb_channels = [39, 19, 9]
    elif dataset_name in ['wv3', 'gf2', 'qb']:
        image_size = 512 if full_res else 256
        image_n_channel = 8 if dataset_name == 'wv3' else 4
        pan_channel = 1
        rgb_channels = [4, 2, 0] if dataset_name == 'wv3' else [2, 1, 0]
    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=pan_channel,#1,
        inner_channel=32,  # 32,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)
    denoise_fn = model_load(weight_path, denoise_fn, device=device)

    denoise_fn.eval()
    print(f"load weight {weight_path}")
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    # load dataset
    d_test = h5py.File(test_data_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        ds_test = PanDataset(
            d_test, full_res=full_res, norm_range=False, division=division, wavelets=True
        )
    else:
        ds_test = HISRDataSets(
            d_test, normalize=False, aug_prob=0.0, wavelets=True
        )
        
    dl_test = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
        )

    saved_name = "reduced" if not full_res else "full"

    # do sampling
    preds = []
    sample_times = len(dl_test)
    analysis = AnalysisPanAcc() if not full_res else NonAnalysisPanAcc()
    for i, batch in enumerate(dl_test):
        if full_res:
            pan, lms, wavelets = batch
            gt = None
        else:
            pan, lms, gt, wavelets = batch
        print(f"sampling [{i}/{sample_times}]")
        pan, lms, wavelets = map(lambda x: x.cuda(), (pan, lms, wavelets))
        cond, _ = einops.pack(
            [lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear")],
            "b * h w",
        )
        sr = diffusion(cond, mode="ddim_sample", section_counts="ddim25")
        sr = sr + lms.cuda()
        sr = sr.clip(0, 1)

        analysis(sr.detach().cpu(), gt)

        print(analysis.print_str(analysis.last_acc))

        if show:
            # hr = hr.detach().clone()[:64]
            # hr = tv.utils.make_grid(hr, nrow=2, padding=0)
            s = tv.utils.make_grid(sr[:64], nrow=4, padding=0).cpu()

            # s = torch.cat([hr, s], dim=-1)  # [b, c, h, 2*w]
            s.clip_(0, 1)
            fig, ax = plt.subplots(
                figsize=(s.shape[2] // 100, s.shape[1] // 100), dpi=200
            )
            ax.imshow(s.permute(1, 2, 0).detach().numpy()[..., rgb_channels])
            ax.set_axis_off()

            # plt.tight_layout()
            # plt.show()
            fig.savefig(
                path_legal_checker(
                    f"./samples/pandiff_{dataset_name}_test/test_sample_iter_{n_steps}_{saved_name}_part_{i}.png"
                ),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )

        sr = sr.detach().cpu().numpy()  # [b, c, h, w]
        sr = sr * division  # [0, 2047]
        preds.append(sr.clip(0, division))
        # preds.append(sr)

        print(f"over all test acc:\n {analysis.print_str()}")

    # save results
    if not full_res:
        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=d_test["gt"][:],
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    else:
        d = dict(
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    model_iterations = weight_path.split("_")[-1].strip(".pth")
    savemat(
        path_legal_checker(f"./samples/mat/test_iter_{model_iterations}_{saved_name}_{dataset_name}.mat"), 
        d
    )
    print("save result")


torch.cuda.set_device(0)
engine_google(
    "YOUR DATA PATH HERE",
    dataset_name="wv3",
    show_recon=True,
    lr_d=1e-4,
    n_steps=500,
    schedule_type="cosine",
    batch_size=64,
    device="cuda:0",
    max_iterations=300_000,
    image_n_channel=4,
    # pretrain_iterations=0,
)

torch.cuda.set_device(0)
test_fn(
    "YOUR DATA PATH HERE",
    batch_size=1,
    n_steps=25, #500,
    show=True,
    dataset_name="gf2",
    division=1023.0,
    full_res=False,
    device="cuda:1",
)
