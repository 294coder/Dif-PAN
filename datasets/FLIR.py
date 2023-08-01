from typing import Union

import h5py
import torch
import torch.nn.functional as F
import torch.utils.data as data


class FLIRDataset(data.Dataset):
    def __init__(self, d: Union[dict, h5py.File]):
        self.d = d
        self.vis = torch.from_numpy(d["data"][:, 0:1].astype("float32"))  # rgb
        self.ir = torch.from_numpy(d["data"][:, 1:].astype("float32"))  # ir
        self.gt = torch.from_numpy(d["data"][:].astype("float32"))  # gt
        self.ms = F.interpolate(
            self.ir, scale_factor=1 / 4, mode="bilinear", align_corners=False
        )
        print('{:^20}{:^20}{:^20}{:^20}'.format('vis', 'ir', 'ms', 'gt'))
        print('{:^20}{:^20}{:^20}{:^20}'.format(str(tuple(self.vis.shape)),
                                                str(tuple(self.ir.shape)),
                                                str(tuple(self.ms.shape)),
                                                str(tuple(self.gt.shape))))

    def __getitem__(self, index):
        """
        output:
            vis [1, 64, 64] domain 1
            ms [1, 16, 16] downsampled ir 1/4
            ir [1, 64, 64] domain 2
            gt [2, 64, 64] 2 domains which give constrains
        Args:
            index: int

        Returns:

        """
        return self.vis[index], self.ms[index], self.ir[index], self.gt[index]

    def __len__(self):
        return self.ir.shape[0]


if __name__ == "__main__":
    from utils import h5py_to_dict

    path = "/home/ZiHanCao/datasets/FLIR/train_vis_ir.h5"
    vis_ir = h5py.File(path, "r")
    d = h5py_to_dict(vis_ir)

    dataset = FLIRDataset(d)
    dl = data.DataLoader(dataset, batch_size=1, shuffle=True)
    for vis, ms, lr, gt in dl:
        print(vis.shape, ms.shape, lr.shape, gt.shape)
        break
