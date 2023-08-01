import numpy
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils.misc import to_numpy
from functools import partial
from .misc import dict_to_str
from ._metric_legacy import analysis_accu


class NonAnalysis(object):
    def __init__(self):
        self.acc_ave = {}  # only used as attribution
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return 'NonAnalysis()'


# FIXME: this python code is not same as matlab code, you should use matlab code to get the real accuracy
# only used in training and validate
class AnalysisPanAcc(object):
    def __init__(self, ergas_ratio: int = 4):
        # metric functions
        self.__sam_ergas_psnr_cc_one_image = partial(analysis_accu, ratio=ergas_ratio, choices=5)
        self.ssim = ssim_batch_tensor_metric

        # tracking accuracy
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = {'SAM': 0., 'ERGAS': 0., 'PSNR': 0., 'CC': 0., 'SSIM': 0.}

    @staticmethod
    def permute_dim(*args, permute_dims=(1, 2, 0)):
        l = []
        for i in args:
            l.append(i.permute(*permute_dims))
        return l

    @staticmethod
    def _sum_acc(d_ave, d_now, n, n2=1):
        assert len(d_ave) == len(d_now)
        for k in d_ave.keys():
            v2 = d_now[k] * n2
            d_ave[k] *= n
            d_ave[k] += v2.cpu().item() if isinstance(v2, torch.Tensor) else v2
        return d_ave

    @staticmethod
    def _average_acc(d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave

    def sam_ergas_psnr_cc_batch(self, gt, pred):
        n = gt.shape[0]
        # input shape should be [B, C, H, W]
        acc_ds = {'SAM': 0., 'ERGAS': 0., 'PSNR': 0., 'CC': 0.}
        for i, (img1, img2) in enumerate(zip(gt, pred)):
            img1, img2 = self.permute_dim(img1, img2)
            acc_d = self.__sam_ergas_psnr_cc_one_image(img1, img2)
            acc_ds = self._sum_acc(acc_ds, acc_d, i)
            acc_ds = self._average_acc(acc_ds, i + 1)
        return acc_ds

    def once_batch_call(self, b_gt, b_pred):
        acc_d1 = self.sam_ergas_psnr_cc_batch(b_gt, b_pred)
        acc_ssim = self.ssim(b_gt, b_pred)
        acc_d1['SSIM'] = acc_ssim
        self._acc_d = acc_d1
        return acc_d1

    def __call__(self, b_gt, b_pred):
        n = b_gt.shape[0]
        self.acc_ave = self._sum_acc(self.acc_ave, self.once_batch_call(b_gt, b_pred), self._call_n, n2=n)
        self.acc_ave = self._average_acc(self.acc_ave, self._call_n + n)
        self._call_n += n
        return self.acc_ave

    def print_str(self):
        return dict_to_str(self.acc_ave)


def normalize_to_01(x):
    # normalize tensor to [0, 1]
    if isinstance(x, torch.Tensor):
        x -= x.flatten(-2).min(-1, keepdim=True)[0][..., None]
        x /= x.flatten(-2).max(-1, keepdim=True)[0][..., None]
    elif isinstance(x, numpy.ndarray):
        x -= x.min((-2, -1), keepdims=True)
        x /= x.max((-2, -1), keepdims=True)
    return x


def psnr_one_img(img_gt, img_test):
    """
    calculate PSNR for one image
    :param img_gt: ground truth image, numpy array, shape [H, W, C]
    :param img_test: test or inference image, numpy array, shape [H, W, C]
    :return: PSNR, float type
    """
    assert img_gt.shape == img_test.shape, 'image 1 and image 2 should have the same size'
    return peak_signal_noise_ratio(img_gt, img_test)


def psnr_batch_tensor_metric(b_gt, b_test):
    """
    calculate PSNR for batch tensor images
    :param b_gt: tensor, shape [B, C, H, W]
    :param b_test: tensor, shape [B, C, H, W]
    :return:
    """
    assert b_gt.shape[0] == b_test.shape[0]
    bs = b_gt.shape[0]
    psnr = 0.
    for gt, t in zip(b_gt, b_test):
        psnr += psnr_one_img(*(to_numpy(gt, t)))
    return psnr / bs


def ssim_one_image(img_gt, img_test, channel_axis=0):
    assert img_gt.shape == img_test.shape, 'image 1 and image 2 should have the same size'
    return structural_similarity(img_gt, img_test, channel_axis=channel_axis)


def ssim_batch_tensor_metric(b_gt, b_test):
    assert b_gt.shape[0] == b_test.shape[0]
    bs = b_gt.shape[0]
    ssim = 0.
    for gt, t in zip(b_gt, b_test):
        ssim += ssim_one_image(*(to_numpy(gt, t)), channel_axis=0)
    return ssim / bs
