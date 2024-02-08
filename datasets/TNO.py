import functools
import os.path
from glob import glob
from typing import Sequence

import PIL.Image as Image
import kornia.augmentation as K
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms


class WindowBasedPadder(object):
    def __init__(self, window_size=64) -> None:
        self.window_size = window_size
        self.padding_fn = None

    def find_least_pad(self, base_size: tuple, window_size: int):
        least_size = []
        for b_s in base_size:
            if b_s % window_size == 0:
                least_size.append(b_s)
            else:
                mult = b_s // window_size
                mult += 1
                least_size.append(mult * window_size)
        return least_size

    def __call__(self, img: torch.Tensor, size: Sequence[int] = None, no_check_pad: bool = False):
        if no_check_pad:
            assert self.padding_fn is not None
            return self.padding_fn(img)

        if size is not None:
            self._last_img_ori_size = size
            self.padding_fn = K.PadTo(size)
        else:
            pad_size = self.find_least_pad(img.shape[-2:], self.window_size)
            self._last_img_ori_size = img.shape[-2:]
            self.padding_fn = K.PadTo(pad_size)

        return self.padding_fn(img)

    def inverse(self, img: torch.Tensor):
        return self.padding_fn.inverse(img, size=self._last_img_ori_size)


class TNODataset(Data.Dataset):
    """
    Infrared dataset

    inter output:
        vis, ms_vis, ir, gt(cat[ir, vis])
    """

    def __init__(
            self, base_dir: str, mode: str, size: int = 64, no_split=False, aug_prob=0.0
    ):
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.base_dir = base_dir
        self.no_split = no_split
        self.size = size

        if mode == "train":
            mode = "training_data"
            # mode = 'new_training_data'
            # print('TNO dataset, using new training data')
            infrared_name = "ir"
            vis_name = "vi"
        else:  # test
            mode = "test_data"
            infrared_name = "ir"
            vis_name = "vi"

        self.infrared_paths = sorted(
            glob(base_dir + f"/{mode}/{infrared_name}/*"), key=os.path.basename
        )
        self.vis_paths = sorted(
            glob(base_dir + f"/{mode}/{vis_name}/*"), key=os.path.basename
        )

        print(f"{mode} file - num of files: {len(self.infrared_paths)}, size {size}")

        to_tensor = transforms.ToTensor()

        self.ir_imgs = []
        self.vis_imgs = []
        for ir_p, vi_p in zip(self.infrared_paths, self.vis_paths):
            ir_img = Image.open(ir_p)
            vi_img = Image.open(vi_p)

            ir_img = self.check_convert(ir_img)
            vi_img = self.check_convert(vi_img)

            # print(ir_img.size, vi_img.size)
            if ir_img.size != vi_img.size:
                print(
                    os.path.basename(ir_p),
                    os.path.basename(vi_p),
                    ir_img.size,
                    vi_img.size,
                )
                minimum_size = (
                    vi_img.size if vi_img.size[0] < ir_img.size[0] else vi_img.size
                )

                ir_img = ir_img.resize(minimum_size)
            self.ir_imgs.append(to_tensor(ir_img))
            self.vis_imgs.append(to_tensor(vi_img))

        self.gt = [
            torch.cat([vis, ir], dim=0) for ir, vis in zip(self.ir_imgs, self.vis_imgs)
        ]

        self.random_crop_ori = K.AugmentationSequential(
            K.RandomCrop(size=(size, size), pad_if_needed=True, keepdim=True),
            # K.Normalize([0.5], [0.5], keepdim=True),
            data_keys=["input", "input", "input"],
        )
        self.down_sample = functools.partial(
            torch.nn.functional.interpolate, scale_factor=1 / 4, mode="bilinear"
        )
        if aug_prob == 0.0:
            self.random_aug_K = None
        else:
            self.random_aug_K = K.AugmentationSequential(
                K.RandomVerticalFlip(p=aug_prob, keepdim=True),
                K.RandomHorizontalFlip(p=aug_prob, keepdim=True),
                # K.RandomRotation(degrees=(-15, 15), p=aug_prob, keepdim=True),
                K.RandomBoxBlur(p=aug_prob / 4, keepdim=True),
                K.RandomSharpness(0.2, p=aug_prob, keepdim=True),
                # K.RandomPlasmaContrast(roughness=(0.2, 0.6), p=aug_prob, keepdim=True),
                # K.RandomPlasmaBrightness(p=aug_prob, keepdim=True),
                data_keys=["input", "input", "input"],
            )
        self.random_crop_ms = transforms.RandomCrop(size // 4)

    def check_convert(self, x: Image.Image):
        if len(x.mode) > 2:
            x, *_ = x.convert("YCbCr").split()
            # print('debug: convert ycbcr')
        else:  # only gray
            x = x.convert("L")  # this is not needed, but keep anyway
        return x

    def __len__(self):
        return len(self.ir_imgs)

    def process_data(self, *imgs):
        processed_imgs = self.random_crop_ori(*imgs)
        if self.random_aug_K is not None:
            processed_imgs = self.random_aug_K(*processed_imgs)
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

    dataset = TNODataset(
        r"/home/ZiHanCao/datasets/RoadScene_and_TNO/",
        "test",
        no_split=True,
        aug_prob=0.0,
    )
    dl = Data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)

    img_padder = WindowBasedPadder(64)

    h_max = 0
    w_max = 0
    for i, (ir, ms, vis, gt) in enumerate(dl):
        # print(ir.min(), vis.min(), gt.min())
        # print(ir.max(), vis.max(), gt.max())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()

        print(ir.shape)
        ir_pad = img_padder(ir)

        inv_ir = img_padder.inverse(ir_pad)

        axes[0].imshow(ir[0].permute(1, 2, 0), "gray")
        axes[1].imshow(ir_pad[0].permute(1, 2, 0), "gray")
        axes[2].imshow(inv_ir[0].permute(1, 2, 0), "gray")

        fig.savefig(f'./pad_{i}.png', dpi=200, bbox_inches='tight')

        # grid = gridspec.GridSpec(2, 2)
        # axes = [
        #     plt.subplot(grid[0, 0]),
        #     plt.subplot(grid[0, 1]),
        #     plt.subplot(grid[1, 0]),
        #     plt.subplot(grid[1, 1]),
        # ]
        # axes[0].imshow(ir[0].permute(1, 2, 0), "gray")
        # axes[0].set_title("ir")
        # axes[1].imshow(ms[0].permute(1, 2, 0), "gray")
        # axes[1].set_title("ms_vis")
        # axes[2].imshow(vis[0].permute(1, 2, 0), "gray")
        # axes[2].set_title("vis")
        # # only show channel 3 image
        # axes[3].imshow(torch.cat([gt[0], gt[0, 0:1]], dim=0).permute(1, 2, 0))
        # axes[3].set_title("gt")
        # plt.show()
        # # fig = plt.gcf()
        # # fig.savefig(f'./{i}.png')
        #
        if i > 4:
            break
