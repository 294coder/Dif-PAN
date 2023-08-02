# from model.dcformer_mwsa import DCFormerMWSA
from model.dcformer_mwsa_wx import DCFormerMWSA
from model.module.MoGFusionHead import MoGFusionHead as FusionHeadWarpper
from model.base_model import BaseModel, register_model, PatchMergeModule

import torch
import torch.nn as nn
import torch.nn.functional as F


@register_model("dcformer_mwsa_mog_fusion_head")
class DCFormerMWSAMoGFusionHead(DCFormerMWSA):
    def __init__(
        self,
        # spectral_num,
        # mode="SUM",
        # channel_list=(64, (32, 64), (32, 64, 128)),
        # block_list=(1, (1, 1), (1, 1, 1)),  # (4, (4, 3), (4, 3, 2)),
        # added_c=1,
        # residual=True,
        spatial_size,
        spectral_num,
        mode="SUM",
        channel_list=(64, (32, 64), (32, 64, 128)),
        block_list=(1, (1, 1), (1, 1, 1)),  # (4, (4, 3), (4, 3, 2)),
        num_heads=(2, (2, 2), (2, 2, 2)),
        mlp_ratio=(2, (2, 2), (2, 2, 2)),
        attn_drop=0.2,
        drop_path=0.2,
        mlp_drop=0.0,
        added_c=1,
        norm_type="bn",
        residual=True,
        patch_merge_step=False,
        patch_size_list=[64, 64, 16],
        crop_batch_size=20,
        scale=8,
        *,
        pretrain_path="/Data2/ZiHanCao/exps/panformer/weight/dcformer_7u5y5qpi.pth",
    ):
        super().__init__(
            spatial_size,
            spectral_num,
            mode,
            channel_list,
            block_list,
            num_heads,
            mlp_ratio,
            attn_drop,
            drop_path,
            mlp_drop,
            added_c,
            norm_type,
            residual,
            patch_merge_step=False,  # always set to False
            patch_size_list=patch_size_list,
            crop_batch_size=crop_batch_size,
            scale=scale,
        )
        self.dcformer = super()
        if pretrain_path is not None:
            self.dcformer.load_state_dict(
                torch.load(pretrain_path, map_location="cuda")["model"]
            )
            print("MoGFusionHead: load pretrain model from", pretrain_path)
        self.body = FusionHeadWarpper(self.dcformer)
        if patch_merge_step:
            self._patch_merge_model = PatchMergeModule(
                # net=self,
                patch_merge_step=self.patch_merge_step,
                crop_batch_size=crop_batch_size,
                patch_size_list=patch_size_list,
                scale=scale,
            )

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self.body(pan, lms, ms)
        loss = criterion(sr, gt)

        return sr, loss

    def val_step(self, ms, lms, pan):
        # or lr_hsi, hr_hsi, rgb)
        mms = F.interpolate(
            ms,
            size=(lms.size(-1) // 2, lms.size(-1) // 2),
            mode="bilinear",
            align_corners=True,
        )
        win_sizes = list(self.window_dict_train_reduce.values())
        chop_list = self.patch_list
        # 16, 32, 64, 64
        self._set_window_dict(
            {
                chop_list[0]: win_sizes[2],
                chop_list[1]: win_sizes[1],
                chop_list[2]: win_sizes[0],
                chop_list[3]: win_sizes[0],
            }
        )
        sr = self._patch_merge_model.forward_chop(ms, mms, lms, pan)[0]
        self._set_window_dict(self.window_dict)
        return sr

    def patch_merge_step(self, ms, mms, lms, pan, **kwargs):
        # all shape is 64
        # mms = F.interpolate(
        #     lms,
        #     size=(split_size // 2, split_size // 2),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        # ms = F.interpolate(
        #     lms,
        #     size=(split_size // 4, split_size // 4),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        # if hisi:
        #     pan = pan[:, :3]
        # else:
        #     pan = pan[:, :1]

        # sr = self.body(pan, lms, ms)

        # return sr
        sr = self.body(pan, lms, ms)  # sr[:,[29,19,9]]
        return sr


if __name__ == "__main__":
    ms = torch.randn(1, 31, 128, 128).cuda(1)
    mms = torch.randn(1, 31, 256, 256).cuda(1)
    lms = torch.randn(1, 31, 512, 512).cuda(1)
    pan = torch.randn(1, 3, 512, 512).cuda(1)

    # net = DCFormerMWSAMoGFusionHead(31, "C", added_c=3, block_list=[4, [4, 3], [4, 3, 2]]).cuda()
    net = DCFormerMWSAMoGFusionHead(
        64,
        31,
        "C",
        added_c=3,
        channel_list=[48, [48, 96], [48, 96, 192]],
        num_heads=(8, (8, 8), (8, 8, 8)),
        mlp_ratio=(2, (2, 2), (2, 2, 2)),
        attn_drop=0.0,
        drop_path=0.0,
        block_list=[1, [1, 1], [1, 1, 1]],
        norm_type="ln",
        patch_merge_step=True,
        patch_size_list=[
            16,
            32,
            64,
            64,
        ],  # [32, 128, 256, 256],  # [200, 200, 100, 25],
        scale=4,
        crop_batch_size=18,
    ).cuda(1)

    print(net.val_step(ms, lms, pan).shape)
