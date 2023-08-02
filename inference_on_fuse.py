import numpy as np
import torch as th
import torch
import os

from model.build_network import build_network
from datasets.TNO import TNODataset
from datasets.FLIR_2 import FLIRDataset
from datasets.TNO import WindowBasedPadder
from utils import AnalysisFLIRAcc

# path = "/Data2/ZiHanCao/datasets/TNO"
# ds = TNODataset(path, "test", no_split=True)
path = '/Data2/ZiHanCao/datasets/RoadSceneFusion_1'
ds = FLIRDataset(path, 'test', no_split=True)
dl = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

os.makedirs('/Data2/ZiHanCao/exps/panformer/visualized_img/ir_RS', exist_ok=True)
print('make dir:', '/Data2/ZiHanCao/exps/panformer/visualized_img/ir_RS')


device = torch.device("cuda:1")
torch.cuda.set_device(device)

net = build_network(
    "dcformer_mwsa_new",
    spatial_size=64,
    spectral_num=1,
    added_c=1,
    mode="C",
    channel_list=[8, [8, 16], [8, 16, 24]],  # [48, [48, 96], [48, 96, 192]]  # [24, [24, 48], [24, 48, 96]]
    num_heads=[4, [4, 4], [8, 8, 8]],
    mlp_ratio=[1, [1, 1], [1, 1, 1]],  # [2, [2, 2], [2, 2, 2]]
    block_list=[2, [2, 2], [2, 2, 2]],
    norm_type="ln",
    patch_merge_step=True,
    patch_size_list=[16, 32, 64, 64],  # [32, 128, 256, 256] #[64, 64, 32, 8]
    scale=4,
    attn_drop=0.2,
    drop_path=0.2,
    crop_batch_size=1,
    residual=True,
    # spectral_num=1,
    # added_c=1,
    # block_list=[1, [1, 1], [1, 1, 1]],
    # mode="C",
    # residual=False,
)
net.load_state_dict(
    # th.load('/home/ZiHanCao/exps/panformer/weight/dcformer_379zkf3e/ep_550.pth', map_location=device)['model']  # 2n8eo45b
    # th.load('./weight/dcformer_17rgbfmz/ep_490.pth', map_location=device)['model']
    # th.load("./weight/dcformer_1qxd4k6t/ep_620.pth", map_location=device)["model"]  # mgimci loss
    # th.load("./weight/dcformer_1at6je5u/ep_1280.pth", map_location=device)["model"]  # pia loss
    # th.load("./weight/dcformer_10yvuovr/ep_1300.pth", map_location=device)["model"]  # pia loss + lpips loss
    # th.load("./weight/dcformer_10m1crxr/ep_920.pth", map_location=device)['model']  # pia loss + lpips loss and TNO new training dataset
    # th.load("./weight/dcformer_3m7v12ou/ep_200.pth", map_location=device)['model']  # TNO and RoadScence dataset toghter training
    # th.load("./weight/dcformer_2705u7eg/ep_1160.pth", map_location=device)['model']  # u2fusion original loss, not work, patch artifact
    # th.load("./weight/dcformer_332f31gf/ep_720.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_13g4q0ls/ep_440.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_34jj20so/ep_60.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_1beukp0d.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_391xl8vy.pth", map_location=device)["model"]
    th.load("./weight/dcformer_13xe8kzx.pth", map_location=device)["model"]  # dcformer wx new arch (residual vis ir mean)

    # ydtr
    # th.load('./weight/ydtr_3gdmei16.pth', map_location=device)['model']
)
base_ws = 16
net = net.cuda()

import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def convert_uint8(img):
    print(img.shape, end=" ")
    if img.dtype != np.uint8:
        img = img.clip(0, 1)
        img *= 255
        # print('convert to [0, 255]')
    return img.astype(np.uint8)


net.eval()
# patch_merge_net = PatchMergeModule(
#     crop_batch_size=1,
#     patch_size_list=[16, 32, 64, 64],
#     net=net,
# )
padder = WindowBasedPadder(64)
ms_padder = WindowBasedPadder(64)
acc_analysiser = AnalysisFLIRAcc()
with th.no_grad():
    for i, (ir, ms, vis, gt) in tqdm(enumerate(dl), total=len(dl)):
        ir, ms, vis, gt = ir.cuda(), ms.cuda(), vis.cuda(), gt.cuda()

        ir = padder(ir)
        vis = padder(vis, no_check_pad=True)
        gt = padder(gt, no_check_pad=True)

        shape = ir.shape[-2:]
        ms_shape = (torch.tensor(shape) // 4).tolist()
        ms = ms_padder(ms, size=ms_shape)

        # spa_size = gt.shape[-2:]
        # pan_nc = ir_RS.size(1)
        # ms_nc = ms.size(1)
        # inp = (
        #     F.interpolate(
        #         ms, size=tuple(vis.shape[-2:]), mode="bilinear", align_corners=True
        #     ),
        #     vis,
        #     torch.cat([ir_RS, torch.zeros(1, ms_nc - pan_nc, *spa_size).cuda()], dim=1),
        # )
        # sr = patch_merge_net.forward_chop(*inp)[0]

        # max_k = ir_RS.shape[-1]

        # window_dict = {}
        # for j in range(3):
        #     window_dict[max_k // (2**j)] = base_ws // (2**j)

        # net._set_window_dict(window_dict)
        sr = net.val_step(ms, vis, ir)
        sr = padder.inverse(sr)
        gt = padder.inverse(gt)

        acc_analysiser(gt, sr)

        sr = sr.detach().cpu().numpy()[0]
        sr_show = sr.transpose([1, 2, 0])
        vis_show = vis.detach().cpu().numpy()[0].transpose([1, 2, 0])
        ir_show = ir.detach().cpu().numpy()[0].transpose([1, 2, 0])

        fig, axes = plt.subplots(ncols=3, figsize=(12, 4), dpi=200)
        axes = axes.flatten()

        for img, name, ax in zip(
                [vis_show, ir_show, sr_show], ["vis", "ir_RS", "fuse"], axes
        ):
            ax.imshow(img, "gray")
            ax.set_axis_off()
            ax.set_title(name)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0)
        plt.show()

        sr_show = convert_uint8(sr_show)
        cv2.imwrite(f"./visualized_img/ir_RS/{i}.bmp", sr_show)
        print("img saved to {}".format(f"./visualized_img/ir_RS/{i}.bmp"))

    print(acc_analysiser.print_str())
