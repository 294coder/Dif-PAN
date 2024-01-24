# import os
# import importlib

from model.build_network import build_network
from model.base_model import MODELS

# ==============================================
# register all models
# from model.DCFNet import DCFNet
# from model.FusionNet import FusionNet
from model.PANNet import VanillaPANNet
# from model.M3DNet import M3DNet
# from model.panformer import PanFormerGAU, PanFormerUNet2, PanFormerSwitch, PanFormerUNet, PanFormer
# from model.dcformer import DCFormer
# from model.dcformer_dpw import DCFormer_DPW
# from model.dcformer_dpw_woo import DCFormer_DPW_WOO
# from model.dcformer_dynamic import DCFormerDynamicConv
# from model.dcformer_reduce import DCFormer_Reduce
# from model.dcformer_mwsa import DCFormerMWSA
# from model.dcformer_mwsa_wx import DCFormerMWSA
# from model.fuseformer import MainNet
# from model.dcformer_reduce_c_64 import DCFormer_Reduce_C64
# from model.dcformer_reduce_c_32_tmp import DCFormer_Reduce_C32
# from model.dcformer_sg_c32 import DCFormer_SG_C32
# from model.dcformer_mobile_x8 import DCFormerMobile
# from model.ydtr import MODEL as YDTR
# from model.CSSNet import Our_netf
# from model.mmnet import MMNet
# from model.pmacnet import PMACNet

from model.LFormer import AttnFuseMain
from model.LFormer_cvpr_re import AttnFuseMain


# from model.gppnn_cvpr import GPPNN
# from model.hypertransformer import HyperTransformer
# from model.hypertransformer import HyperTransformerPre

# ablation
# from model.dcformer_abla_only_channel_attn import DCFormer_XCA
# from model.dcformer_abla_only_mwa import DCFormerOnlyMWA
# from model.dcformer_abla_only_cross_branch_mwsa import DCFormerOnlyCrossBranchMWSA

# disscussion
# from model.dcformer_disscuss_mog_fusion_head import DCFormerMWSAMoGFusionHead

# others
# from model.GPPNN import GPPNN

# import os
# import importlib
# file_p = os.path.dirname(__file__)
# model_ps = os.listdir(file_p)
# model_ps.remove('base_model.py')
# model_ps.remove('__init__.py')

# _all_models = [
#     importlib.import_module('model.' + p[:-3])
#     for p in model_ps if p.endswith('.py')
# ]
# ==============================================
