import time
from typing import Union
import PIL.Image
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as T
import pywt
import cv2
import numpy as np
import h5py
import random
from typing import List, Tuple, Optional


class Identity:
    def __call__(self, *args):
        # args is a tuple
        # return is also a tuple
        return args


class HISRDataSets(data.Dataset):
    # FIXME: when use this Dataset, you should set num_works to 0 or it will raise unpickable error
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        normalize=False,
        aug_prob=0.0,
        wavelets=False,
    ):
        super(HISRDataSets, self).__init__()
        # warning: you should not save file (h5py.File) in this class,
        # or it will raise CAN NOT BE PICKLED error in multiprocessing
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        if isinstance(file, (str, h5py.File)):
            if isinstance(file, str):
                file = h5py.File(file)
            print(
                "warning: when @file is a h5py.File object, it can not be pickled.",
                "try to set DataLoader number_worker to 0",
            )
        assert not normalize, '@normalize should be False'
        
        self.gt, self.lr_hsi, self.rgb, self.hsi_up = self._split_parts(file, normalize)

        self.wavelets = wavelets
        if wavelets:
            print("processing wavelets...")
            hsi_up_main, (hsi_h, hsi_v, hsi_d) = pywt.wavedec2(
                self.hsi_up, wavelet="db1", level=1, axes=(-2, -1)
            )
            rgb_main, (rgb_h, rgb_v, rgb_d) = pywt.wavedec2(
                self.rgb, wavelet="db1", level=1, axes=(-2, -1)
            )
            print("done.")
            self.wavelet_dcp = np.concatenate(
                [hsi_up_main, rgb_h, rgb_v, rgb_d], axis=1
            )

        self.size = self.gt.shape[-2:]
        print("dataset shape:")
        print("{:^20}{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb", "gt"))
        print(
            "{:^20}{:^20}{:^20}{:^20}".format(
                str(tuple(self.lr_hsi.shape)),
                str(tuple(self.hsi_up.shape)),
                str(tuple(self.rgb.shape)),
                str(tuple(self.gt.shape)),
            )
        )
        # geometrical transformation
        self.aug_prob = aug_prob
        self.geo_trans = (
            T.Compose(
                [
                    # T.RandomHorizontalFlip(p=self.aug_prob),
                    # T.RandomVerticalFlip(p=self.aug_prob),
                    T.RandomApply(
                        [
                            T.RandomErasing(
                                p=self.aug_prob, scale=(0.02, 0.15), ratio=(0.2, 1.0)
                            ),
                            T.RandomAffine(
                                degrees=(0, 70),
                                translate=(0.1, 0.2),
                                scale=(0.95, 1.2),
                                interpolation=T.InterpolationMode.BILINEAR,
                            ),
                        ],
                        p=self.aug_prob,
                    ),
                    # T.RandomAutocontrast(p=self.aug_prob),
                    # T.RandomAdjustSharpness(sharpness_factor=2, p=self.aug_prob)
                    # T.RandomErasing(p=self.aug_prob)
                ]
            )
            if aug_prob != 0.0
            else Identity()
        )

    def _split_parts(self, file, normalize=False, load_all=True):
        # has already been normalized
        if load_all:
            # load all data in memory
            if normalize:
                return (
                    np.array(file["GT"][:], dtype=np.float32) / 2047.0,
                    np.array(file["LRHSI"][:], dtype=np.float32) / 2047.0,
                    np.array(file["RGB"][:], dtype=np.float32) / 2047.0,
                    np.array(file["HSI_up"][:], dtype=np.float32) / 2047.0,
                )
            else:
                return (
                    np.array(file["GT"][:], dtype=np.float32),
                    np.array(file["LRHSI"][:], dtype=np.float32),
                    np.array(file["RGB"][:], dtype=np.float32),
                    np.array(file["HSI_up"][:], dtype=np.float32),
                )
        else:
            # warning: it will ignore @normalize
            return (
                file.get("GT"),
                file.get("LRHSI"),
                file.get("RGB"),
                file.get("HSI_up"),
            )

    def aug_trans(self, *data):
        data_list = []
        seed = torch.random.seed()
        for d in data:
            torch.manual_seed(seed)
            random.seed(seed)
            d = self.geo_trans(d)
            data_list.append(d)
        return tuple(data_list)

    def __getitem__(self, index):
        # gt: [31, 64, 64]
        # lr_hsi: [31, 16, 16]
        # rbg: [3, 64, 64]
        # hsi_up: [31, 64, 64]

        # harvard [rgb]
        # cave [bgr]
        if self.wavelets:
            tuple_data = (
                self.rgb[index],
                # self.lr_hsi[index],
                self.hsi_up[index],
                self.gt[index],
                self.wavelet_dcp[index],
            )
        else:
            tuple_data = (
                self.rgb[index],
                # self.lr_hsi[index],
                self.hsi_up[index],
                self.gt[index],
            )
        if self.aug_prob != 0.0:
            return self.aug_trans(*tuple_data)
        else:
            return tuple_data

    def __len__(self):
        return len(self.gt)


if __name__ == "__main__":
    path = r"/home/ZiHanCao/datasets/HISI/new_cave/x8/validation_cave(with_up)x8_rgb.h5"
    file = h5py.File(path)
    dataset = HISRDataSets(file, aug_prob=0.9)
    dl = data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )
    for i, (rgb, lr_hsi, hsi_up, gt) in enumerate(dl, 1):
        print(
            f"lr_hsi: {lr_hsi.shape}, rgb: {rgb.shape}, hsi_up: {hsi_up.shape}, gt: {gt.shape}"
        )
        fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
        axes[0].imshow(rgb[0].permute(1, 2, 0).numpy()[..., :3])
        axes[1].imshow(lr_hsi[0].permute(1, 2, 0).numpy()[..., :3])
        axes[2].imshow(hsi_up[0].permute(1, 2, 0).numpy()[..., :3])
        axes[3].imshow(gt[0].permute(1, 2, 0).numpy()[..., :3])
        plt.tight_layout(pad=0)
        plt.show()
        time.sleep(3)
        # fig.savefig(f'./tmp/{i}.png', dpi=100)
