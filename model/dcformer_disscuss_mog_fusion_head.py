from model.dcformer_mwsa import DCFormerMWSA
from model.module.MoGFusionHead import MoGFusionHead as FusionHeadWarpper
from model.base_model import BaseModel, register_model

import torch
import torch.nn as nn
import torch.nn.functional as F


@register_model("dcformer_mwsa_mog_fusion_head")
class DCFormerMWSAMoGFusionHead(DCFormerMWSA):
    def __init__(
        self,
        spectral_num,
        mode="SUM",
        channel_list=(64, (32, 64), (32, 64, 128)),
        block_list=(1, (1, 1), (1, 1, 1)),  # (4, (4, 3), (4, 3, 2)),
        added_c=1,
        residual=True,
        *,
        pretrain_path='./weight/dcformer_1dpmi7w6/ep_30.pth',
    ):
        super().__init__(
            spectral_num, mode, channel_list, block_list, added_c, residual
        )
        self.dcformer = super()
        if pretrain_path is not None:
            self.dcformer.load_state_dict(torch.load(pretrain_path, map_location="cuda")['model'])
            print('MoGFusionHead: load pretrain model from', pretrain_path)
        self.body = FusionHeadWarpper(self.dcformer)
        
    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self.body(pan, lms, ms)
        loss = criterion(sr, gt)

        return sr, loss

    def val_step(self, ms, lms, pan):
        # or lr_hsi, hr_hsi, rgb)
        sr = self.body(pan, lms, ms)
        return sr
    
    def patch_merge_step(self, ms, lms, pan, hisi=False, split_size=64):
        # all shape is 64
        # mms = F.interpolate(
        #     lms,
        #     size=(split_size // 2, split_size // 2),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        ms = F.interpolate(
            lms,
            size=(split_size // 4, split_size // 4),
            mode="bilinear",
            align_corners=True,
        )
        if hisi:
            pan = pan[:, :3]
        else:
            pan = pan[:, :1]

        sr = self.body(pan, lms, ms)

        return sr
    
    
if __name__ == '__main__':
    ms = torch.randn(1, 31, 16, 16).cuda()
    mms = torch.randn(1, 31, 32, 32).cuda()
    lms = torch.randn(1, 31, 64, 64).cuda()
    pan = torch.randn(1, 3, 64, 64).cuda()
    
    net = DCFormerMWSAMoGFusionHead(31, "C", added_c=3, block_list=[4, [4, 3], [4, 3, 2]]).cuda()
    
    print(
        net.val_step(ms, lms, pan).shape
    )
