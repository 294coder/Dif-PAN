import time
from typing import Union
import PIL.Image
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import h5py
import random
from typing import List, Tuple, Optional


class HISRDataSets(data.Dataset):
    # FIXME: when use this Dataset, you should set num_works to 0 or it will raise unpickable error
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        normalize=False,
        aug_prob=0.0,
        rgb_to_bgr=False,
        full_res=False
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
        self.full_res = full_res
        data_s= self._split_parts(
            file, normalize, rgb_to_bgr=rgb_to_bgr, full=full_res
        )
        
        if len(data_s) == 4:
            self.gt, self.lr_hsi, self.rgb, self.hsi_up = data_s

        else:
            self.lr_hsi, self.rgb, self.hsi_up = data_s           
        
        self.size = self.rgb.shape[-2:]
        print("dataset shape:")

        # print dataset info
        if not full_res:
            print("{:^20}{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb", "gt"))
            print(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
                    str(tuple(self.gt.shape)),
                )
            )
            
        else:
            print("{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb"))
            print(
                "{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
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
            else Indentity()
        )

    def _split_parts(self, file, load_all=True, rgb_to_bgr=False, keys=None, full=False):
        # has already been normalized
        
        
        # warning: key RGB is HRMSI when the dataset is GF5-GF1
        if not full:
            keys = ['GT', 'LRHSI', 'HRMSI', 'HSI_up']
        else:
            keys = ['LRHSI', 'HRMSI', 'HSI_up']
        
        if load_all:
            # load all data in memory
            data = []
            for k in keys:
                data.append(
                    torch.tensor(file[k][:], dtype=torch.float32),
                )
            if rgb_to_bgr:
                print("warning: rgb to bgr, for testing generalization only.")
                # rgb -> bgr
                if not full:
                    data[2] = data[2][:, [-1, 1, 0]]
                else:
                    data[1] = data[1][:, [-1, 1, 0]]
            return data
        else:
            # warning: it will ignore @normalize
            # warning: "GT" can not be access in FULL mode
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
        if not self.full_res:
            tuple_data = (
                self.rgb[index],
                self.lr_hsi[index],
                self.hsi_up[index],
                self.gt[index],
            )
        else:
            tuple_data = (
                self.rgb[index],
                self.lr_hsi[index],
                self.hsi_up[index],
            )
        if self.aug_prob != 0.0:
            return self.aug_trans(*tuple_data)
        else:
            return tuple_data

    def __len__(self):
        return len(self.rgb)


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
