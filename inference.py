from functools import partial

import h5py
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from scipy.io import savemat
from datasets.TNO import TNODataset

from datasets.wv3 import WV3Datasets
from datasets.HISR import HISRDataSets
from datasets.gf import GF2Datasets
from datasets.FLIR_2 import FLIRDataset
from model import build_network
from utils import (
    AnalysisPanAcc,
    viz_batch,
    yaml_load,
    res_image,
    unref_for_loop,
    ref_for_loop,
    config_py_load,
    module_load
)
from utils.visualize import invert_normalized

device = "cuda:1"
torch.cuda.set_device(device)
dataset_type = "gf"
save_format = "mat"
full_res = False
split_patch = True
patch_size = 16
ergas_ratio = 4
patch_size_list = [
    patch_size // ergas_ratio,
    patch_size // 2,
    patch_size,
    patch_size,
]  # ms, lms, pan
save_mat = True
loop_func = (
    partial(
        ref_for_loop,
        hisi=dataset_type in ["cave", "cave_x8", "harvard", "harvard_x8", "gf5"],
        patch_size_list=patch_size_list,
        ergas_ratio=ergas_ratio,
        residual_exaggerate_ratio=5000,
    )
    if not full_res
    else partial(
        unref_for_loop,
        hisi=dataset_type in ["cave", "cave_x8", "harvard", "harvard_x8", "gf5"],
        patch_size_list=patch_size_list,
    )
)
name = "lformer"
subarch = ""
dl_bs = 1
crop_bs = 2


print("=" * 50)
print("dataset:", dataset_type)
print("model:", name + (("_" + subarch) if subarch else ""))
print("full_res:", full_res)
if split_patch:
    print("patch_size:", patch_size)
    print("patch size list:", patch_size_list)

print("=" * 50)

# ======================worldview3 checkpoint============
## old datsets
# p = './weight/pannet_3vskemk0.pth'  # pannet
# p = './weight/panformer_44v9p9t3.pth'  # gau
# p = './weight/panformer_1bxe3b0v.pth'  # restormer
# p = './weight/fusionnet_30zvejpu.pth'  # fusionnet
# p = './weight/dcformer_2t9tw637.pth'  # dcformer with attention grad
# p = './weight/dcformer_22vv4nzx.pth'  # dcformer sota
# p = './weight/dcformer_2vt2nluz.pth'  # dcformer_dpw without attention grad
# p = './weight/dcformer_1thsgpdv.pth'  # dcformer_dpw with partially switch qkv
# p = './weight/dcformer_3jhs05so.pth'  # dcformer_dpw with fully switch qkv
# p = './weight/dcformer_3vg8vlh9.pth'  # dcformer_dpw with only switch qkv in fuse layer, depth = [4, [4, 4], [4, 4, 4]] and another cross-attn in FPN between b1_in and b2_in(after CFSBlock)
# p = './weight/dcformer_2azisub2.pth'  # dcformer with less depth (all depth is 2)
# p = './weight/dcformer_3baebtne.pth'  # dcformer_dpw [4, [4, 3], [4, 3, 2]]
# p = './weight/dcformer_16hbiq2c.pth'  # dcformer_woo [4, [4, 3], [4, 3, 2]]
# p = './weight/dcformer_19e6v9x5.pth'  # dcformer_reduce [k,s,p]=[4,4,2]
# p = './weight/dcformer_lvu3ts9m.pth'  # dcformer_reduce [k,s,p]=[5,4,2], feature maps are reduced to [16/4, 32/4, 64/4]
# p = './weight/dcformer_o7woscjm.pth'  # dcfomer use avgpool
# p = ./weight/dcformer_2sxj5ebt.pth'  # dcformer use residual in attention and l1 train, 600 epochs
# p = './weight/dcformer_15lj5cdt.pth'  # dcformer use conv for reduce k, v spatial size, 600 epochs
# p = './weight/dcformer_2enz0i9d.pth'  # dcformer dynamic kernel

## new datsets
# p = './weight/dcfnet_2ui2wcqd.pth'  # dcfnet without lms added
# p = './weight/dcfnet_2c67s082.pth'  # dcfnet add lms
# p = './weight/dcfnet_2qg57pin.pth'  # dcfnet
# p = './weight/dcformer_fyw69uxb.pth'  # dcformer mwsa blocklist=(4, (4, 3), (4, 3, 2))
# p = './weight/dcformer_hbnwhpe8/ep_230.pth'  # dcformer_mwsa (r)

# p = './weight/dcformer_1g9ljhul.pth'  # dcformer_mwsa wx 8 CAttn

# p = "./weight/hpmnet_kqv7vcpy.pth"  # HMPNet

# p = "./weight/lformer_16nzc16d.pth"  # lformer ablation (skip attention)
# p = "./weight/lformer_dcu45ddw.pth"  # lformer

# ========================================================

# ================HISI CAVE checkpoint=============
##### cave_x4
# p = './weight/dcformer_37xwviyh.pth'  # dcformer_reduce on HISI CAVE dataset
# p = './weight/dcformer_cr61n8zi.pth'  # dcformer_mwsa block_list=(4, (4, 4), (4, 4, 4)) PSNR: 50.206
# p = './weight/dcformer_2rzb1pf3.pth'
# p = './weight/dcformer_3dioh24t.pth'  # dcformer_reduce_c32 block_list=(4, (4, 4), (4, 4, 4)) PSNR: 51.50
# p = './weight/dcformer_350s795b.pth'  # dcformer_mwsa block_list=(4, (4, 3), (4, 3, 2)) PSNR:51.24
# p = './weight/dcformer_2jyvj7ac/ep_110.pth'  # dcformer_mwsa with ghost PSNR: 51.35
# p = "./weight/dcformer_1dpmi7w6/ep_30.pth"  # dcformer_mwsa PSNR: 51.39 (r)

# p = "./weight/dcformer_cave_x4.pth"  # dcformer new arch wx
# p = './weight/dcformer_7u5y5qpi.pth'  # dcformer 8 CAttn

####### cave_x8
# p = "./weight/dcformer_15g03tzt.pth"  # 10->80
# p = "./weight/dcformer_3n8ejb6g.pth"  # 16->96
# p = './weight/dcformer_2avwr28h.pth'  # dcformer mwsa 16->96
# p = './weight/dcformer_1dxkpbs2.pth'  # dcformer_mwsa ghost module 16->128 block_list [4, [6, 3], [6, 3, 2]]
# p = "./weight/dcformer_21updqvy/ep_1110.pth"  # dcformer_mwsa (r)

# p = "./weight/dcformer_3e77ot3s.pth"  # dcformer new arch wx c_attn legacy
# p = './weight/dcformer_1gx5sc1l.pth'  # dcformer 8 CAttn

# p = './weight/dcformer_2hwz3dgf.pth'

##### harvard_x8
# p = "./weight/dcformer_zeavxkwx.pth"  # dcformer_mwsa ghost module
# p = './weight/dcformer_22hf3ncx.pth'  # retrain

# p = './weight/dcformer_3rwfkdra.pth'  # dcformer new arch wx c_attn legacy low psnr
# p = './weight/dcformer_dkwinunx.pth'  # dcformer 8 CAttn

#### harvard_x4
# p = './weight/fuseformer_ufsb66w3.pth'  # harvard fuseformer
# p = './weight/dcformer_37o38nol.pth'  # harvard new dataset dcformer better performance
# p = './weight/dcformer_n142idj8.pth'  # dcformer (1, (1, 1), (1, 1, 1))
# p = './weight/dcformer_1mju6inh.pth'  # dcformer_c32 longer training
# p = './weight/dcformer_2n5poffy.pth'  # dcformer_mwsa PSNR: 48.67

# p = './weight/dcformer_ko3dx5dh/ep_100.pth'  # dcformer_mwsa (r)

# p = './weight/dcformer_3pexzxle.pth'  # dcformer new arch wx c_attn legacy low psnr
# p = './weight/dcformer_3esy9p4b.pth'  # dcformer 8 CAttn
# =================================================

# ============== GF5-GF1 ==========================
# p = './weight/hsrnet_mvqqs7jp.pth'  # HSRNet

# p = 'weight/fuseformer_3gq75ygm.pth'

# ===============GF checkpoint=====================
# p = './weight/dcformer_2a8g853d.pth'  # dcformer
# p = './weight/dcformer_d9hhb681.pth'  # dcformer_mwsa
# p = './weight/dcformer_2sn1ox21.pth'  # dcformer_mwsa 1/6
# p = './weight/dcformer_3b0qmez5/ep_90.pth'  # dcformer_mwsa (r)
# p = './weight/dcformer_1vhkamhc.pth'  # dcformer new arch wx
# p = './weight/dcformer_sv76u5vk.pth'  # dcformer 8 CAttn

# p = './weight/dcfnet_2k1xjqom.pth'  # dcfnet

# p = './weight/mmnet_frw0mwwn.pth'  # mmnet
# p = './weight/pmacnet_y2k8paq1.pth'  # pmacnet

# p = './weight/hpmnet_2re44fdd/ep_600.pth'

p = './weight/lformer_3dvlsog6.pth'
# =================================================

# ===============QB checkpoint=====================
# p = './weight/dcformer_1h6bfguv.pth'  # dcformer_mwsa new train 1/3
# p = './weight/dcformer_3hmfhwn5.pth'  # dcformer_mwsa ft on new train 1/4(better reduce, worse full dataset)
# p = './weight/dcformer_387tvha8/ep_310.pth'  # dcformer_mwsa (r)

# p = './weight/dcfnet_3m7y84a3.pth'  # dcfnet
# p = './weight/dcfnet_l5r28gfz.pth'  # lr=1e-4
# p = './weight/dcformer_xig9pbqs.pth' # dcformer 8CAttn


# p = './weight/pannet_3knmo9wy.pth'  # pannet

# p = "./weight/hpmnet_3vgc0ov9.pth"  # hpmnet
# =================================================

# ==============FLIR checkpoint===================
# p = './weight/dcformer_37ovgekt/ep_1200.pth'  # RoadScence residual False

# p = './weight/dcformer_17rgbfmz/ep_490.pth'  # TNO residual True

# p = './weight/dcfomer_f313brtd.pth' # RS dcformer wx new
# ================================================

# ==================ablation study===================
# p = './weight/dcformer_3bxaqe7l.pth'  # dcformer only xca
# p = './weight/dcformer_3q5m6cx4.pth'  # dcformer only mwa
# p = './weight/dcformer_2iwd2pk0/ep_5.pth'  # dcformer_mwsa without ghost module
# p='./weight/dcformer_2skeztu4.pth'  # dcformer only cross-branch mwsa

# p = './weight/dcfnet_2d85ivqq.pth'  # dcfnet baseline

# p = './weight/dcformer_imejxfph.pth'  # dcformer_mwsa window_size 64:32
# p = './weight/dcformer_2y6zpd1n.pth'  # dcformer_mwsa window_size 64:8

# p = './weight/dcformer_2sqn5nyx.pth'  # 8/4/2
# p = './weight/dcformer_g4k7ob6g.pth'  # 32/16/8


# cross-scale xca tab. 7 row 2
# p = './weight/dcformer_1scc05fk.pth'

# cross-scale mwsa tab.7 row 3
# p = './weight/dcformer_28e1kx3c.pth'
# p = './weight/dcformer_2bbwni53.pth'

# in-scale mwsa tab.7 row 4
# p = './weight/dcformer_1yb9gy7x.pth'
# p = './weight/dcformer_22cv4bb7.pth'

# ====================================================

# ==================discussion study===================
# cave x4
# dcformer mog fusion head
# p = './weight/dcformer_1beukp0d.pth'

# dcformer CSNLN fusion head
# p = './weight/dcformer_317n1zsw.pth'

# =====================================================


if dataset_type == "wv3":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/wv3/reduced_examples/test_wv3_multiExm1.h5"
    else:
        # path = '/home/ZiHanCao/datasets/pansharpening/wv3/full_examples/test_wv3_OrigScale_multiExm1.h5'
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5"
elif dataset_type == "cave":
    path = "/Data2/ZiHanCao/datasets/HISI/new_cave/test_cave(with_up)x4.h5"
elif dataset_type == "cave_x8":
    path = "/Data2/ZiHanCao/datasets/HISI/new_cave/x8/test_cave(with_up)x8_rgb.h5"
elif dataset_type == "harvard":
    # path = "/Data2/ZiHanCao/datasets/HISI/new_harvard/test_harvard(with_up)x4_rgb.h5"
    path = "/Data2/ShangqiDeng/data/HSI/harvard_x4/test_harvard(with_up)x4_rgb200.h5"
elif dataset_type == "harvard_x8":
    path = "/Data2/ZiHanCao/datasets/HISI/new_harvard/x8/test_harvard(with_up)x8_rgb.h5"
elif dataset_type == "gf5":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_23tap_new.h5"
    else:
        path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_OrigScale.h5"
elif dataset_type == "gf":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/gf/reduced_examples/test_gf2_multiExm1.h5"
    else:
        # path = '/home/ZiHanCao/datasets/pansharpening/gf/full_examples/test_gf2_OrigScale_multiExm1.h5'
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_gf2_OrigScale_multiExm1.h5"
elif dataset_type == "qb":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/qb/reduced_examples/test_qb_multiExm1.h5"
    else:
        # path = '/home/ZiHanCao/datasets/pansharpening/qb/full_examples/test_qb_OrigScale_multiExm1.h5'
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_qb_OrigScale_multiExm1.h5"
elif dataset_type == "wv2":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/wv2/reduced_examples/test_wv2_multiExm1.h5"
    else:
        # path = '/home/ZiHanCao/datasets/pansharpening/wv2/full_examples/test_wv2_OrigScale_multiExm1.h5'
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv2_OrigScale_multiExm1.h5"
elif dataset_type == "roadscene":
    path = "/Data2/ZiHanCao/datasets/RoadSceneFusion_1"
elif dataset_type == "tno":
    path = "/Data2/ZiHanCao/datasets/TNO"
else:
    raise NotImplementedError("not exists {} dataset".format(dataset_type))

# model = VanillaPANNet(8, 32).to('cuda:0')

config = yaml_load(name)
if name in ["panformer", "dcformer", "lformer"]:
    full_arch = name + "_" + subarch if subarch != "" else name
    model = build_network(full_arch, **config["network_configs"][full_arch])
else:
    model = build_network(name, **config["network_configs"])

# -------------------load params-----------------------
# params = torch.load(p, map_location=device)
# odict = OrderedDict()
# for k, v in params['model'].items():
#    odict['module.' + k] = v


# model.load_state_dict(params["model"])
model = module_load(p, model, device, strict=True)
model = model.to(device)
model.eval()
# -----------------------------------------------------

# -------------------get dataset-----------------------
if dataset_type in ["wv3", "qb", "wv2"]:
    d = h5py.File(path)
    ds = WV3Datasets(d, hp=False, full_res=full_res)
elif dataset_type in ["cave", "harvard", "cave_x8", "harvard_x8", "gf5"]:
    d = h5py.File(path)
    ds = HISRDataSets(d, full_res=full_res)
elif dataset_type == "gf":
    d = h5py.File(path)
    ds = GF2Datasets(d, full_res=full_res)
elif dataset_type == "roadscene":
    ds = FLIRDataset(path, "test", no_split=False)
elif dataset_type == "tno":
    ds = TNODataset(path, "test", no_split=False)
else:
    raise NotImplementedError
dl = data.DataLoader(ds, batch_size=dl_bs)
# -----------------------------------------------------

# -------------------inference-------------------------
all_sr = loop_func(model, dl, device, crop_bs, split_patch=split_patch)
# -----------------------------------------------------

# -------------------save result-----------------------
d = {}
# FIXME: there is an error here, const should be 1023. when sensor is gf
if dataset_type in ["wv3", "qb", "wv2"]:
    const = 2047.0
elif dataset_type in ["gf"]:
    const = 1023.0
elif dataset_type in [
    "cave",
    "harvard",
    "cave_x8",
    "harvard_x8",
    "roadscene",
    "tno",
    "gf5",
]:
    const = 1.0
else:
    raise NotImplementedError
cat_sr = np.concatenate(all_sr, axis=0).astype("float32")
d["sr"] = np.asarray(cat_sr) * const
try:
    d["gt"] = np.asarray(ds.gt[:]) * const
except:
    print("no gt")
    pass

if save_mat:  # torch.tensor(d['sr'][:, [4,2,0]]),  torch.tensor(d['gt'][:, [4,2,0]])
    _ref_or_not_s = "unref" if full_res else "ref"
    _patch_size_s = f"_p{patch_size}" if split_patch else ""
    if dataset_type not in [
        "cave",
        "harvard",
        "cave_x8",
        "harvard_x8",
        "gf5",
    ]:  # wv3, qb, gf
        d["ms"] = np.asarray(ds.ms[:]) * const
        d["lms"] = np.asarray(ds.lms[:]) * const
        d["pan"] = np.asarray(ds.pan[:]) * const
    else:
        d["ms"] = np.asarray(ds.lr_hsi[:]) * const
        d["lms"] = np.asarray(ds.hsi_up[:]) * const
        d["pan"] = np.asarray(ds.rgb[:]) * const

    if save_format == "mat":
        path = f"./visualized_img/data_{name}{subarch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.mat"
        savemat(path, d)
    else:
        path = f"./visualized_img/data_{name}{subarch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.h5"
        save_file = h5py.File(path, "w")
        save_file.create_dataset("sr", data=d["sr"])
        save_file.create_dataset("ms", data=d["ms"])
        save_file.create_dataset("lms", data=d["lms"])
        save_file.create_dataset("pan", data=d["pan"])
        save_file.close()
    print(f"save results in {path}")
# -----------------------------------------------------
