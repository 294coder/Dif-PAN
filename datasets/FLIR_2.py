import functools
from glob import glob

import PIL.Image as Image
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.nn import Identity


class FLIRDataset(Data.Dataset):
    """
    Infrared dataset

    inter output:
        vis, ms_vis, ir, gt(cat[ir, vis])
    """

    def __init__(self, base_dir: str, mode: str, size: int = 64, no_split=False):
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.base_dir = base_dir
        self.no_split = no_split

        if mode == "train":
            infrared_name = "infrared"
            vis_name = "visible"
            suffix = 'jpg'
            crop = True
            sort_fn = lambda x: x
        else:
            infrared_name = "ir test"
            vis_name = "vi test"
            suffix = 'bmp'
            crop = True
            sort_fn = lambda x: sorted(x, key=lambda y: int(y.split('/')[-1].split('.')[0]))

        self.infrared_paths = sort_fn(glob(base_dir + f"/{mode}/{infrared_name}/*.{suffix}"))
        self.vis_paths = sort_fn(glob(base_dir + f"/{mode}/{vis_name}/*.{suffix}"))

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

        self.random_crop_ori = transforms.RandomCrop(size) if crop else Identity()
        self.down_sample = functools.partial(
            torch.nn.functional.interpolate, scale_factor=1 / 4, mode="bilinear"
        )
        # self.random_crop_ms = transforms.RandomCrop(size // 4)

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

    dataset = FLIRDataset(r"/Data2/ZiHanCao/datasets/RoadSceneFusion_1", "test")
    dl = Data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (ir, ms, vis, gt) in enumerate(dl):
        # print(ir.max(), vis.max(), gt.max())
        print(ir.shape, ms.shape, vis.shape, gt.shape)

        # grid = gridspec.GridSpec(2, 2)
        # axes = [
        #     plt.subplot(grid[0, 0]),
        #     plt.subplot(grid[0, 1]),
        #     plt.subplot(grid[1, 0]),
        #     plt.subplot(grid[1, 1]),
        # ]
        # axes[0].imshow(vis[0].permute(1, 2, 0), "gray")
        # axes[0].set_title("ir")
        # axes[1].imshow(ms[0].permute(1, 2, 0), "gray")
        # axes[1].set_title("ms_vis")
        # axes[2].imshow(ir[0].permute(1, 2, 0), "gray")
        # axes[2].set_title("vis")
        # # only show channel 3 image
        # axes[3].imshow(torch.cat([gt[0], gt[0, 0:1]], dim=0).permute(1, 2, 0))
        # axes[3].set_title("gt")
        # plt.show()

        if i > 10:
            break
