from calendar import c
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import numpy as np
from math import exp
import lpips

# from utils.torch_dct import dct_2d, idct_2d
from utils._vgg import vgg16
from utils._ydtr_loss import ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi


class PerceptualLoss(nn.Module):
    def __init__(self, percep_net="vgg", norm=True):
        super(PerceptualLoss, self).__init__()
        self.norm = norm
        self.lpips_loss = lpips.LPIPS(net=percep_net).cuda()

    def forward(self, x, y):
        # assert x.shape == y.shape
        loss = self.lpips_loss(x, y, normalize=self.norm)
        return torch.squeeze(loss).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class MaxGradientLoss(torch.nn.Module):
    def __init__(self, mean_batch=True) -> None:
        super().__init__()
        self.register_buffer(
            "x_sobel_kernel",
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(1, 1, 3, 3),
        )
        self.register_buffer(
            "y_sobel_kernel",
            torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).expand(1, 1, 3, 3),
        )
        self.mean_batch = mean_batch

    def forward(self, fuse, ir, vis):
        c = fuse.size(1)

        fuse_grad_x = F.conv2d(fuse, self.x_sobel_kernel, padding=1, groups=c)
        fuse_grad_y = F.conv2d(fuse, self.y_sobel_kernel, padding=1, groups=c)

        ir_grad_x = F.conv2d(ir, self.x_sobel_kernel, padding=1, groups=c)
        ir_grad_y = F.conv2d(ir, self.y_sobel_kernel, padding=1, groups=c)

        vis_grad_x = F.conv2d(vis, self.x_sobel_kernel, padding=1, groups=c)
        vis_grad_y = F.conv2d(vis, self.y_sobel_kernel, padding=1, groups=c)

        max_grad_x = torch.maximum(ir_grad_x, vis_grad_x)
        max_grad_y = torch.maximum(ir_grad_y, vis_grad_y)

        if self.mean_batch:
            max_gradient_loss = (
                F.l1_loss(fuse_grad_x, max_grad_x) + F.l1_loss(fuse_grad_y, max_grad_y)
            ) / 2
        else:
            x_loss_b = F.l1_loss(fuse_grad_x, max_grad_x, reduction="none").mean(
                dim=(1, 2, 3)
            )
            y_loss_b = F.l1_loss(fuse_grad_y, max_grad_y, reduction="none").mean(
                dim=(1, 2, 3)
            )

            max_gradient_loss = (x_loss_b + y_loss_b) / 2

        return max_gradient_loss


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def mci_loss(pred, gt):
    return F.l1_loss(pred, gt.max(1, keepdim=True)[0])


def sf(f1, kernel_radius=5):
    """copy from https://github.com/tthinking/YDTR/blob/main/losses/__init__.py

    Args:
        f1 (torch.Tensor): image shape [b, c, h, w]
        kernel_radius (int, optional): kernel redius using calculate sf. Defaults to 5.

    Returns:
        loss: loss item. type torch.Tensor
    """

    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = (
        torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    b_shift_kernel = (
        torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(
        F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1
    )
    return 1 - f1_sf


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridSSIMSF(torch.nn.Module):
    def __init__(self, channel, weighted_r=(1.0, 5e-2, 6e-4, 25e-5)) -> None:
        super().__init__()
        self.weighted_r = weighted_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = ssim_loss_ir(fuse, ir)
        _ssim_f_vi = ssim_loss_vi(fuse, vi)
        _sf_f_ir = sf_loss_ir(fuse, ir)
        _sf_f_vi = sf_loss_vi(fuse, vi)

        ssim_f_ir = self.weighted_r[0] * _ssim_f_ir
        ssim_f_vi = self.weighted_r[1] * _ssim_f_vi
        sf_f_ir = self.weighted_r[2] * _sf_f_ir
        sf_f_vi = self.weighted_r[3] * _sf_f_vi

        loss_dict = dict(
            ssim_f_ir=ssim_f_ir,
            ssim_f_vi=ssim_f_vi,
            sf_f_ir=sf_f_ir,
            sf_f_vi=sf_f_vi,
        )

        loss = ssim_f_ir + ssim_f_vi + sf_f_ir + sf_f_vi
        return loss, loss_dict


class HybridSSIMMCI(torch.nn.Module):
    def __init__(self, channel, weight_r=(1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.ssim = SSIMLoss(channel=channel)
        self.mci_loss = mci_loss
        self.weight_r = weight_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = self.weight_r[0] * self.ssim(fuse, ir)
        _ssim_f_vi = self.weight_r[1] * self.ssim(fuse, vi)
        _mci_loss = self.weight_r[2] * self.mci_loss(fuse, gt)

        loss = _ssim_f_ir + _ssim_f_vi + _mci_loss

        loss_dict = dict(
            ssim_f_ir=_ssim_f_ir,
            ssim_f_vi=_ssim_f_vi,
            mci_loss=_mci_loss,
        )

        return loss, loss_dict


def accum_loss_dict(ep_loss_dict: dict, loss_dict: dict):
    for k, v in loss_dict.items():
        if k in ep_loss_dict:
            ep_loss_dict[k] += v
        else:
            ep_loss_dict[k] = v
    return ep_loss_dict


def ave_ep_loss(ep_loss_dict: dict, ep_iters: int):
    for k, v in ep_loss_dict.items():
        ep_loss_dict[k] = v / ep_iters
    return ep_loss_dict


def ave_multi_rank_dict(rank_loss_dict: list[dict]):
    ave_dict = {}
    n = len(rank_loss_dict)
    assert n >= 1, "@rank_loss_dict must have at least one element"
    keys = rank_loss_dict[0].keys()

    for k in keys:
        vs = 0
        for d in rank_loss_dict:
            v = d[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            vs = vs + v
        ave_dict[k] = vs / n
    return ave_dict


class HybridL1SSIM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1)):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return (loss,)


class HybridMCGMCI(torch.nn.Module):
    def __init__(self, weight_r=(1.0, 1.0)) -> None:
        super().__init__()
        self.mcg = MaxGradientLoss()
        self.mci = mci_loss
        self.weight_r = weight_r

    def forward(self, pred, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        mcg_loss = self.mcg(pred, ir, vis) * self.weight_r[0]
        mci_loss = self.mci(pred, gt) * self.weight_r[1]

        loss_dict = dict(mcg=mcg_loss, mci=mci_loss)

        return mcg_loss + mci_loss, loss_dict


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter2 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter1.weight.data = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )
    filter2.weight.data = (
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


class LossWarpper(torch.nn.Module):
    def __init__(self, weighted_ratio=(1.0, 1.0), **losses):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(losses.keys())
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt) -> tuple[torch.Tensor, dict[torch.Tensor]]:
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            loss += l
            d_loss[n] = l
        return loss, d_loss


class TorchLossWrapper(torch.nn.Module):
    def __init__(self, weight_ratio: Union[tuple[float], list[float]], **loss) -> None:
        super().__init__()
        self.key = list(loss.keys())
        self.loss = list(loss.values())
        self.weight_ratio = weight_ratio

        assert len(weight_ratio) == len(loss.keys())

    def forward(self, pred, gt):
        loss_total = 0.0
        loss_d = {}
        for i, l in enumerate(self.loss):
            loss_i = l(pred, gt) * self.weight_ratio[i]
            loss_total = loss_total + loss_i

            k = self.key[i]
            loss_d[k] = loss_i

        return loss_total, loss_d


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class RMILoss(nn.Module):
    def __init__(
        self,
        with_logits=False,
        radius=3,
        bce_weight=0.5,
        downsampling_method="max",
        stride=3,
        use_log_trace=True,
        use_double_precision=True,
        epsilon=0.0005,
    ):

        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):

        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

        return bce

    def rmi_loss(self, input, target):

        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ self.transpose(y)
        p_cov = p @ self.transpose(p)
        y_p_cov = y @ self.transpose(p)

        m = y_cov - y_p_cov @ self.transpose(
            self.inverse(p_cov + eps)
        ) @ self.transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * self.log_trace(m + eps)
        else:
            rmi = 0.5 * self.log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == "region-extraction" else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius**2, -1))
        return x_regions

    def downsample(self, x):

        if self.stride == 1:
            return x

        if self.downsampling_method == "region-extraction":
            return x

        padding = self.stride // 2
        if self.downsampling_method == "max":
            return F.max_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        if self.downsampling_method == "avg":
            return F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        raise ValueError(self.downsampling_method)

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def inverse(x):
        return torch.inverse(x)

    @staticmethod
    def log_trace(x):
        x = torch.linalg.cholesky(x)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)

    @staticmethod
    def log_det(x):
        return torch.logdet(x)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()


class HybridSSIMRMIFuse(nn.Module):
    def __init__(self, weight_ratio=(1.0, 1.0), ssim_channel=1):
        super().__init__()
        # self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channel=ssim_channel)
        self.rmi = RMILoss(bce_weight=0.6)
        self.weight_ratio = weight_ratio

    def forward(self, fuse, x):
        fuse = fuse.clip(0, 1)

        vis = x[:, 0:1]
        ir = x[:, 1:]

        ssim_loss = self.ssim(fuse, vis) + self.ssim(fuse, ir)
        rmi_loss = self.rmi(fuse, vis) + self.rmi(fuse, ir)

        loss_d = dict(ssim=ssim_loss, rmi=rmi_loss)

        loss = self.weight_ratio[0] * ssim_loss + self.weight_ratio[1] * rmi_loss
        return loss, loss_d


class HybridPIALoss(nn.Module):
    def __init__(self, weight_ratio=(3, 7, 20, 10)) -> None:
        super().__init__()
        assert (
            len(weight_ratio) == 4
        ), "@weight_ratio must be a tuple or list of length 4"
        self.weight_ratio = weight_ratio
        self._mcg_loss = MaxGradientLoss()
        self.perceptual_loss = PerceptualLoss(norm=True)

    def forward(self, fuse, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        l1_int = (F.l1_loss(fuse, vis) + F.l1_loss(fuse, ir)) * self.weight_ratio[0]
        l1_aux = (F.l1_loss(fuse, gt.max(1, keepdim=True)[0])) * self.weight_ratio[1]

        # FIXME: this should implement as the largest gradient of vis and ir_RS
        # l1_grad = (F.l1_loss(gradient(fuse), gradient(vis)) + F.l1_loss(gradient(fuse), gradient(ir_RS))) * \
        #           self.weight_ratio[2]
        l1_grad = self._mcg_loss(fuse, ir, vis) * self.weight_ratio[2]
        percep_loss = (
            self.perceptual_loss(fuse, vis) + self.perceptual_loss(fuse, ir)
        ) * self.weight_ratio[3]

        loss_d = dict(
            intensity_loss=l1_int,
            context_loss=l1_aux,
            gradient_loss=l1_grad,
            percep_loss=percep_loss,
        )

        return l1_int + l1_aux + l1_grad + percep_loss, loss_d


# U2Fusion dynamic loss weight
class U2FusionLoss(nn.Module):
    def __init__(self, loss_weights: tuple[float] = (5, 2, 10)) -> None:
        # loss_weights:
        super().__init__()
        # modified from https://github.com/ytZhang99/U2Fusion-pytorch/blob/master/train.py
        # and https://github.com/linklist2/PIAFusion_pytorch/blob/master/train_fusion_model.py
        # no normalization
        # so do not unormalize the input

        assert len(loss_weights) == 3, "loss_weights must be a tuple of length 3"

        self.feature_model = vgg16(pretrained=True)
        self.c = 0.1
        self.loss_weights = loss_weights
        self.ssim_loss = SSIMLoss(channel=1)
        #   , size_average=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, fuse, gt):
        # similiar to PIAFusion paper, which introduces a classifier
        # to judge the day or night image and give the probability
        ws = self.dynamic_weight(gt)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        # here we do not follow U2Fusion paper and change it into other losses
        l1_int = (
            vi_w * self.mse_loss(fuse, gt[:, 0:1]).mean((1, 2, 3))
            + ir_w * self.mse_loss(fuse, gt[:, 1:]).mean((1, 2, 3))
        ).mean() * self.loss_weights[0]

        # 哪里亮点哪里
        l1_aux = F.mse_loss(fuse, gt.max(1, keepdim=True)[0]) * self.loss_weights[1]

        # gradient part. choose the largest gradient
        # l1_grad = (
        #         F.l1_loss(
        #             gradient(fuse),
        #             torch.maximum(
        #                 vi_w[:, None, None, None] * gradient(gt[:, 0:1]),
        #                 ir_w[:, None, None, None] * gradient(gt[:, 1:]),
        #             ),
        #         )
        #         * self.loss_weights[2]
        # )

        # l1_grad = (
        #     self.l1_loss(gradient(fuse), vi_w * gradient(gt[:, 0:1])).mean((1, 2, 3))
        #     + self.l1_loss(gradient(fuse), ir_w * gradient(gt[:, 1:])).mean((1, 2, 3))
        # ).mean()

        # ssim loss would cause window artifacts
        # loss_ssim = (
        #     ir_w * self.ssim_loss(fuse, gt[:, 1:])
        #     + vi_w * self.ssim_loss(fuse, gt[:, 0:1])
        # ).mean() * self.loss_weights[2]
        loss_ssim = (
            self.ssim_loss(fuse, gt[:, 0:1]) + self.ssim_loss(fuse, gt[:, 1:])
        ) * self.loss_weights[2]

        loss_d = dict(intensity_loss=l1_int, aux_loss=l1_aux, ssim_loss=loss_ssim)
        # print(ir_w, vi_w)

        return l1_int + l1_aux + loss_ssim, loss_d

    @torch.no_grad()
    def dynamic_weight(self, gt):
        ir_vgg, vi_vgg = self.repeat_dims(gt[:, 1:]), self.repeat_dims(gt[:, 0:1])

        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)
            # if i == 0:
            #     w1 = torch.unsqueeze(m1, dim=-1)
            #     w2 = torch.unsqueeze(m2, dim=-1)
            # else:
            #     w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
            #     w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.c
        weight_2 = torch.mean(w2, dim=-1) / self.c

        # print(weight_1.tolist()[:6], weight_2.tolist()[:6])

        weight_list = torch.stack(
            [weight_1, weight_2], dim=-1
        )  # torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        # _, c, _, _ = features.shape
        # c = int(c)
        # for i in range(c):
        #     feat_grad = F.conv2d(
        #         features[:, i : i + 1, :, :], kernel, stride=1, padding=1
        #     )
        #     if i == 0:
        #         feat_grads = feat_grad
        #     else:
        #         feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) == 1, "the number of channel of x must be 1"
        return x.repeat(1, 3, 1, 1)


# DCT Blur Loss
class DCTBlurLoss(nn.Module):
    def __init__(self, temperature=100, reduction="mean") -> None:
        super().__init__()
        self.t = temperature
        self.reduction = reduction
        self.distance = nn.L1Loss(reduction=reduction)
        if reduction == "none":
            self.feature_model = vgg16(pretrained=True)
            self.c = 0.1

    @staticmethod
    def heat_blur_torch(img, t=25):
        K1 = img.shape[-2]
        K2 = img.shape[-1]

        dct_img = dct_2d(img, norm="ortho")  # [3, K1, K2]
        freqs_h = torch.pi * torch.linspace(0, K1 - 1, K1) / K1  # [K1]
        freqs_w = torch.pi * torch.linspace(0, K2 - 1, K2) / K2  # [K2]

        freq_square = (freqs_h[:, None] ** 2 + freqs_w[None, :] ** 2).to(
            img.device
        )  # [K1, K2]
        dct_img = dct_img * torch.exp(-freq_square[None, ...] * t)  # [3, K1, K2]

        recon_img = idct_2d(dct_img, norm="ortho")

        return recon_img

    def forward(self, f, gt):
        ws = self.dynamic_weight(gt)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        f_dct_blur = self.heat_blur_torch(f, t=self.t)
        vi_dct_blur, ir_dct_blur = self.heat_blur_torch(gt, t=self.t).chunk(2, dim=1)

        f_vi_loss = self.distance(f_dct_blur, vi_dct_blur)
        f_ir_loss = self.distance(f_dct_blur, ir_dct_blur)

        if self.reduction == "none":
            f_vi_loss = f_vi_loss.mean(dim=(1, 2, 3))
            f_ir_loss = f_ir_loss.mean(dim=(1, 2, 3))

            ws = self.dynamic_weight(gt)
            ir_w, vi_w = ws.chunk(2, dim=-1)
            ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

            f_vi_loss = f_vi_loss * vi_w
            f_ir_loss = f_ir_loss * ir_w

        return (f_vi_loss + f_ir_loss).mean()

    @torch.no_grad()
    def dynamic_weight(self, gt):
        ir_vgg, vi_vgg = self.repeat_dims(gt[:, 1:]), self.repeat_dims(gt[:, 0:1])

        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.c
        weight_2 = torch.mean(w2, dim=-1) / self.c
        weight_list = torch.stack([weight_1, weight_2], dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) == 1, "the number of channel of x must be 1"
        return x.repeat(1, 3, 1, 1)

################# SwinFusion loss helper functions #################

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)#.cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)#.cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        image_A = image_A.unsqueeze(0)
        image_B = image_B.unsqueeze(0)      
        intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity
    
################# SwinFusion loss #################

class SwinFusionLoss(nn.Module):
    def __init__(self):
        super(SwinFusionLoss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        
    def forward(self, image_fused, gt):
        image_A = gt[:, 0:1]
        image_B = gt[:, 1:]
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        
        loss_d = {'loss_l1': loss_l1, 'loss_gradient': loss_gradient, 'loss_SSIM': loss_SSIM}
        return fusion_loss, loss_d #loss_gradient, loss_l1, loss_SSIM

#####################################################


class CDDFusionLoss(nn.Module):
    def __init__(self, weights=(1, 1, 1)) -> None:
        super().__init__()
        self.weights = weights

        self.l1_loss = nn.L1Loss()
        self.dct_loss = DCTBlurLoss(reduction="none")
        self.mcg_loss = MaxGradientLoss()

    def forward(self, f, gt):
        l1_loss = self.l1_loss(f, gt.max(dim=1, keepdim=True)[0]) * self.weights[0]
        dct_loss = self.dct_loss(f, gt) * self.weights[1]
        mcg_loss = self.mcg_loss(f, gt[:, 0:1], gt[:, 1:]) * self.weights[2]

        loss_d = dict(l1_loss=l1_loss, dct_loss=dct_loss, mcg_loss=mcg_loss)

        return l1_loss + dct_loss + mcg_loss, loss_d
    
    
def HPM_gradient_diff(Pred, GT):  
    R = F.pad(GT, [0, 1, 0, 0])[:, :, :, 1:] 
    B = F.pad(GT, [0, 0, 0, 1])[:, :, 1:, :]
    dx1, dy1 = torch.abs(R - GT), torch.abs(B - GT)
    dx1[:, :, :, -1], dy1[:, :, -1, :] = 0, 0 
    R = F.pad(Pred, [0, 1, 0, 0])[:, :, :, 1:] 
    B = F.pad(Pred, [0, 0, 0, 1])[:, :, 1:, :]
    dx2, dy2 = torch.abs(R - Pred), torch.abs(B - Pred)
    dx2[:, :, :, -1], dy2[:, :, -1, :] = 0, 0   
    res = torch.abs(dx2-dx1)+torch.abs(dy2-dy1)
    return res.mean()


def get_loss(loss_type, channel=31):
    if loss_type == "mse":
        criterion = TorchLossWrapper((1., ), mse=nn.MSELoss())  # nn.MSELoss()
    elif loss_type == "l1":
        criterion = TorchLossWrapper((1.,), l1=nn.L1Loss())
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'hpm':
        criterion = TorchLossWrapper((1., 0.3), l1=nn.L1Loss(), grad_loss=HPM_gradient_diff)
    elif loss_type == "l1ssim":
        criterion = HybridL1SSIM(channel=channel, weighted_r=(1.0, 0.1))
    elif loss_type == "ssimrmi_fuse":
        criterion = HybridSSIMRMIFuse(weight_ratio=(1, 1), ssim_channel=channel)
    elif loss_type == "pia_fuse":
        # perceptual loss should be less weighted
        criterion = HybridPIALoss(weight_ratio=(3, 7, 20, 10))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(channel=channel, weighted_r=(1.0, 1.0))
    elif loss_type == "ssimsf":
        # YDTR loss
        # not hack weighted ratio
        criterion = HybridSSIMSF(channel=1)
    elif loss_type == "ssimmci":
        criterion = HybridSSIMMCI(channel=1)
    elif loss_type == "mcgmci":
        criterion = HybridMCGMCI(weight_r=(2.0, 1.0))
    elif loss_type == "u2fusion":
        criterion = U2FusionLoss()
    elif loss_type == "cddfusion":
        criterion = CDDFusionLoss(weights=(1.5, 1, 1))
    elif loss_type == "swinfusion":
        criterion = SwinFusionLoss()
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


if __name__ == "__main__":
    # loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    
    loss = get_loss("hpm")
    
    x = torch.randn(1, 31, 64, 64, requires_grad=True)
    y = x + torch.randn(1, 31, 64, 64) / 10
    l = loss(x, y)
    l.backward()
    print(l)
    print(x.grad)

    import PIL.Image as Image

    vi = (
        np.array(
            Image.open(
                "/Data2/ZiHanCao/datasets/RoadScene_and_TNO/training_data/vi/FLIR_05857.jpg"
            ).convert("L")
        )
        / 255
    )
    ir = (
        np.array(
            Image.open(
                "/Data2/ZiHanCao/datasets/RoadScene_and_TNO/training_data/ir_RS/FLIR_05857.jpg"
            ).convert("L")
        )
        / 255
    )

    torch.cuda.set_device("cuda:0")

    vi = torch.tensor(vi)[None, None].float()  # .cuda()
    ir = torch.tensor(ir)[None, None].float()  # .cuda()

    fuse = ((vi + ir) / 2).repeat_interleave(2, dim=0)
    fuse.requires_grad_()
    print(fuse.requires_grad)

    gt = torch.cat((vi, ir), dim=1).repeat_interleave(2, dim=0)

    # fuse_loss = HybridSSIMRMIFuse(weight_ratio=(1.0, 1.0, 1.0), ssim_channel=1)
    # fuse_loss = U2FusionLoss().cuda(1)
    # fuse_loss = HybridPIALoss().cuda(1)
    # fuse_loss = CDDFusionLoss()  # .cuda()
    fuse_loss = SwinFusionLoss()
    loss, loss_d = fuse_loss(fuse, gt)
    loss.backward()
    print(loss)
    print(loss_d)

    print(fuse.grad)

    # mcg_mci_loss = HybridMCGMCI()
    # print(mcg_mci_loss(fuse, gt))
