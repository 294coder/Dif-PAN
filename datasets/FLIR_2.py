import functools

import torch
import torch.utils.data as Data
import os
import numpy as np
import torchvision.transforms as transforms
from glob import glob
import PIL.Image as Image


class FLIRDataset(Data.Dataset):
    """
    Infrared dataset

    inter output:
        vis, ms_vis, ir, gt(cat[ir, vis])
    """

    def __init__(self, base_dir: str, mode: str, size: int = 128, no_split=False):
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.base_dir = base_dir
        self.no_split = no_split

        if mode == "train":
            infrared_name = "infrared"
            vis_name = "visible"
            suffix='jpg'
        else:
            infrared_name = "ir test"
            vis_name = "vi test"
            suffix='bmp'

        self.infrared_paths = glob(base_dir + f"/{mode}/{infrared_name}/*.{suffix}")
        self.vis_paths = glob(base_dir + f"/{mode}/{vis_name}/*.{suffix}")
        
        if mode == 'test':
            key = lambda x: int(os.path.basename(x.strip('.'+suffix)))
            self.vis_paths.sort(key=key)
            self.infrared_paths.sort(key=key)

        to_tensor = transforms.ToTensor()

        self.ir_imgs = [
            to_tensor(Image.open(path).convert("L")) for path in self.infrared_paths
        ]
        self.vis_imgs = [
            to_tensor(Image.open(path).convert("L")) for path in self.vis_paths
        ]

        self.gt = [
            torch.cat([vis, ir], dim=0) for ir, vis in zip(self.ir_imgs, self.vis_imgs)
        ]

        self.random_crop_ori = transforms.RandomCrop(size)
        self.down_sample = functools.partial(
            torch.nn.functional.interpolate, scale_factor=1 / 4, mode="bilinear"
        )
        self.random_crop_ms = transforms.RandomCrop(size // 4)

    def __len__(self):
        return len(self.ir_imgs)

    def process_data(self, *imgs):
        processed_imgs = []
        seed = torch.seed()
        for img in imgs:
            torch.manual_seed(seed)
            img = self.random_crop_ori(img)
            processed_imgs.append(img)
        return processed_imgs

    def __getitem__(self, index):
        # print(
        #     'ir path: {}, vis path: {}'.format(self.infrared_paths[index], self.vis_paths[index])
        # )
        ir = self.ir_imgs[index]
        vis = self.vis_imgs[index]
        gt = self.gt[index]
        if not self.no_split:
            vis, ir, gt = self.process_data(vis, ir, gt)
        return ir, self.down_sample(vis[None])[0], vis, gt


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    dataset = FLIRDataset(r"/media/office-401-remote/Elements SE/cao/ZiHanCao/datasets/RoadSceneFusion_1", "train")
    dl = Data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (ir, ms, vis, gt) in enumerate(dl):
        print(ir.max(), vis.max(), gt.max())

        grid = gridspec.GridSpec(2, 2)
        axes = [
            plt.subplot(grid[0, 0]),
            plt.subplot(grid[0, 1]),
            plt.subplot(grid[1, 0]),
            plt.subplot(grid[1, 1]),
        ]
        axes[0].imshow(vis[0].permute(1, 2, 0), "gray")
        axes[0].set_title("ir")
        axes[1].imshow(ms[0].permute(1, 2, 0), "gray")
        axes[1].set_title("ms_vis")
        axes[2].imshow(ir[0].permute(1, 2, 0), "gray")
        axes[2].set_title("vis")
        # only show channel 3 image
        axes[3].imshow(torch.cat([gt[0], gt[0, 0:1]], dim=0).permute(1, 2, 0))
        axes[3].set_title("gt")
        plt.show()
        
        plt.savefig(f'RS_test_{i}.png')

        if i > 10:
            break
