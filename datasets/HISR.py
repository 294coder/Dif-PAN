import random
import time
from typing import Union
import cv2

import h5py
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as T

from utils import Identity


class HISRDataSets(data.Dataset):
    # FIXME: when use this Dataset, you should set num_works to 0 or it will raise unpickable error
    def __init__(
        self, file: Union[h5py.File, str, dict], normalize=False, aug_prob=0.0, hp=False
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

        _norm_factor = 1.0 if not normalize else 2047.0
        self.gt, self.lr_hsi, self.rgb, self.hsi_up = self._split_parts(
            file, _norm_factor
        )

        if hp:
            self._hp_kernel = torch.ones(
                (5, 5),
                dtype=torch.float32,
            ).view(1, 1, 5, 5)
            self.group_high_pass()

        # NOTE: bgr -> rgb
        # self.rgb = self.rgb[:, [2, 1, 0]]
        # self.lr_hsi[:, [29, 19, 9]] = self.lr_hsi[:, [9, 19, 29]]
        # self.hsi_up[:, [29, 19, 9]] = self.hsi_up[:, [9, 19, 29]]
        # self.gt[:, [29, 19, 9]] = self.gt[:, [9, 19, 29]]

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
                    T.RandomHorizontalFlip(p=self.aug_prob),
                    T.RandomVerticalFlip(p=self.aug_prob),
                    # T.RandomApply(
                    #     [
                    #         T.RandomErasing(
                    #             p=self.aug_prob, scale=(0.02, 0.15), ratio=(0.2, 1.0)
                    #         ),
                    #         T.RandomAffine(
                    #             degrees=(0, 70),
                    #             translate=(0.1, 0.2),
                    #             scale=(0.95, 1.2),
                    #             interpolation=T.InterpolationMode.BILINEAR,
                    #         ),
                    #     ],
                    #     p=self.aug_prob,
                    # ),
                    # T.RandomAutocontrast(p=self.aug_prob),
                    # T.RandomAdjustSharpness(sharpness_factor=2, p=self.aug_prob)
                    # T.RandomErasing(p=self.aug_prob)
                ]
            )
            if aug_prob != 0.0
            else Identity()
        )

    def _get_high_pass(self, data):
        self._hp_kernel: torch.Tensor

        c = data.size(1)  # data[:4, [29,19,9]], hp_d[:4, [29,19,9]] # data[:4], hp_d[:4]
        hp_d = torch.conv2d(
            data,
            weight=self._hp_kernel.repeat_interleave(c, dim=0),
            groups=c,
            stride=1,
            padding=self._hp_kernel.size(-1) // 2,
        )
        data = data - hp_d
        return data

    def group_high_pass(self):
        self.lr_hsi = self._get_high_pass(self.lr_hsi)
        self.hsi_up = self._get_high_pass(self.hsi_up)
        self.rgb = self._get_high_pass(self.rgb)

    def _split_parts(self, file, normalize_factor=1.0, load_all=True):
        # has already been normalized
        if load_all:
            # load all data in memory
            return (
                torch.tensor(file["GT"][:], dtype=torch.float32) / normalize_factor,
                torch.tensor(file["LRHSI"][:], dtype=torch.float32) / normalize_factor,
                torch.tensor(file["RGB"][:], dtype=torch.float32) / normalize_factor,
                torch.tensor(file["HSI_up"][:], dtype=torch.float32) / normalize_factor,
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
        tuple_data = (
            self.rgb[index],
            self.lr_hsi[index],
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
    path = r"/Data2/ZiHanCao/datasets/HISI/new_cave/validation_cave(with_up)x4.h5"
    file = h5py.File(path)
    dataset = HISRDataSets(file, aug_prob=0.0, hp=True)
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
