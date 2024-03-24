import os
import h5py

from utils.misc import h5py_to_dict, NameSpace
from datasets.wv3 import WV3Datasets
from datasets.gf import GF2Datasets
from datasets.HISR import HISRDatasets
from datasets.TNO import TNODataset
from datasets.FLIR_2  import FLIRDataset


def get_fusion_dataset(args: NameSpace):
    if args.split_ratio is not None and args.path is not None and False:  # never reach here
        # FIXME: only support splitting worldview3 datasets
        # Warn: will be decrepated in the next update
        train_ds, val_ds = make_datasets(
            args.path,
            hp=args.hp,
            seed=args.seed,
            aug_probs=args.aug_probs,
            split_ratio=args.split_ratio,
        )
    else:
        if args.dataset == "flir":
            train_ds = FLIRDataset(args.path.base_dir, "train")
            val_ds = FLIRDataset(args.path.base_dir, "test")
        elif args.dataset == "tno":
            train_ds = TNODataset(
                args.path.base_dir, "train", aug_prob=args.aug_probs[0]
            )
            val_ds = TNODataset(args.path.base_dir, "test", aug_prob=args.aug_probs[1])

        elif args.dataset in [
            "wv3",
            "qb",
            "gf2",
            "cave_x4",
            "harvard_x4",
            "cave_x8",
            "harvard_x8",
            "hisi-houston",
        ]:
            # the dataset has already splitted

            # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
            if hasattr(args.path, "train_path") and hasattr(args.path, "val_path"):
                # 旧代码：手动切换数据集路径
                train_path = args.path.train_path
                val_path = args.path.val_path
            else:
                _args_path_keys = list(args.path.__dict__.keys())
                for k in _args_path_keys:
                    if args.dataset in k:
                        train_path = getattr(args.path, f"{args.dataset}_train_path")
                        val_path = getattr(args.path, f"{args.dataset}_val_path")
            assert (
                train_path is not None and val_path is not None
            ), "train_path and val_path should not be None"

            h5_train, h5_val = (
                h5py.File(train_path),
                h5py.File(val_path),
            )
            if args.dataset in ["wv3", "qb"]:
                d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
                train_ds, val_ds = (
                    WV3Datasets(d_train, hp=args.hp, aug_prob=args.aug_probs[0]),
                    WV3Datasets(d_val, hp=args.hp, aug_prob=args.aug_probs[1]),
                )
            elif args.dataset == "gf2":
                d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
                train_ds, val_ds = (
                    GF2Datasets(d_train, hp=args.hp, aug_prob=args.aug_probs[0]),
                    GF2Datasets(d_val, hp=args.hp, aug_prob=args.aug_probs[1]),
                )
            elif args.dataset[:4] == "cave" or args.dataset[:7] == "harvard":
                keys = ["LRHSI", "HSI_up", "RGB", "GT"]
                if args.dataset.split("-")[-1] == "houston":
                    from einops import rearrange
                    
                    # to avoid unpicklable error
                    def permute_fn(x):
                        return rearrange(x, "b h w c -> b c h w")

                    dataset_fn = permute_fn
                else:
                    dataset_fn = None

                d_train, d_val = (
                    h5py_to_dict(h5_train, keys),
                    h5py_to_dict(h5_val, keys),
                )
                train_ds = HISRDatasets(
                    d_train, aug_prob=args.aug_probs[0], dataset_fn=dataset_fn
                )
                val_ds = HISRDatasets(
                    d_val, aug_prob=args.aug_probs[1], dataset_fn=dataset_fn
                )
                # del h5_train, h5_val
        else:
            raise NotImplementedError(f"not support dataset {args.dataset}")
        
        return train_ds, val_ds
        