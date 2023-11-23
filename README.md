# DDIF
<div style="text-align: center;">
  <a href="https://www.arxiv.org/">
    <img src="https://img.shields.io/badge/arXiv-red.svg?style=flat" alt="ArXiv">
  </a>
    <a href="https://arxiv.org/abs/2304.04774">paper</a>
</div>
**Official repository of "Diffusion model with disentangled modulations for sharpening multispectral and hyperspectral images".**

<div style="text-align: center;">
  </a>
    Abstract:
    The denoising diffusion model has received increasing attention in the field of image generation in recent years, thanks to its powerful generation capability. However, diffusion models should be deeply investigated in the field of multi-source image fusion, such as remote sensing pansharpening and multispectral and hyperspectral image fusion (MHIF). 
    In this paper, we introduce a novel {supervised} diffusion model with two conditional modulation modules, specifically designed for the task of multi-source image fusion. 
    These modules mainly consist of a coarse-grained style modulation (CSM) and a fine-grained wavelet modulation (FWM), which aim to disentangle coarse-grained style information and fine-grained frequency information, respectively, thereby generating competitive fused images. Moreover, some essential strategies for the training of the given diffusion model are well discussed, e.g., the selection of training objectives. 
    The superiority of the proposed method is verified compared with recent state-of-the-art (SOTA) techniques by extensive experiments on two multi-source image fusion benchmarks, i.e., pansharpening and MHIF. In addition, sufficient discussions and ablation studies in the experiments are involved to demonstrate the effectiveness of our approach. 
</a>
</div>
News:

- 2023/11/23: **Code will be released soon!**:fire: 


## Quick Overview

<img src="https://raw.githubusercontent.com/294coder/blog_img_bed/main/img3/202311232300466.png" alt="image-20231123230021337" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/294coder/blog_img_bed/main/img3/202311232301434.png" alt="image-20231123230102164" style="zoom: 46%;" />

## Citation

If you find our paper is useful, please consider to cite:

```tex
@article{cao2023ddrf,
  title={Ddrf: Denoising diffusion model for remote sensing image fusion},
  author={Cao, ZiHan and Cao, ShiQi and Wu, Xiao and Hou, JunMing and Ran, Ran and Deng, Liang-Jian},
  journal={arXiv preprint arXiv:2304.04774},
  year={2023}
}
```

