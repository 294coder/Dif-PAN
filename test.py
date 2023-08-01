import PIL.Image as pim
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from sailency import LAM
import h5py
from model.build_network import build_network
from datasets.HISR import HISRDataSets
from functools import wraps
from model.base_model import PatchMergeModule

device = torch.device("cuda:0")
torch.cuda.set_device(device)


def remove_batch_one_to_np(tensor: torch.Tensor):
    assert tensor.ndim == 4 and tensor.size(0) == 1
    tensor = torch.squeeze(tensor, dim=0).permute(1, 2, 0)  # [h,w,c]
    return tensor.detach().cpu().numpy()

'''
def val_step(model, lms, rgb):
    # @wraps
    # def inner(lr):

     #   return sr

    return model.val_step(lr, lms, rgb)
'''

# CAVE dataset
net = build_network(
    "dcformer_mwsa",
    spectral_num=31,
    added_c=3,
    block_list=[4, [4, 3], [4, 3, 2]],
    mode="C",
)
net.load_state_dict(
    torch.load("./weight/dcformer_1dpmi7w6/ep_30.pth", map_location=device)["model"]
)
torch.cuda.set_device(0)
net = net.cuda()
net.eval()

# dataloader
path = "/home/ZiHanCao/datasets/HISI/new_cave/test_cave(with_up)x4.h5"
bs = 1
d = h5py.File(path)
index = 2
rgb = torch.from_numpy(d["RGB"][index : index + 1].astype("float32"))#.cuda()
ms = torch.from_numpy(d["LRHSI"][index : index + 1].astype("float32"))#.cuda()
lms = torch.from_numpy(d["HSI_up"][index : index + 1].astype("float32"))#.cuda()
hr = d["GT"][index].astype("float32")


# model_fn = val_step(net, lms, rgb)
LAM(net, hr, [lms, ms, rgb], 200, 240, fold=1, window_size=16)
