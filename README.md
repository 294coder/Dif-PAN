# DDIF: Diffusion model with disentangled modulations for sharpening multispectral and hyperspectral images

<div style="text-align: center;">
  <a href="https://www.arxiv.org/">
    <img src="https://img.shields.io/badge/arXiv-red.svg?style=flat" alt="ArXiv">
  </a>
    <a href="https://arxiv.org/abs/2304.04774">Arxiv</a>
    <img class="gh-logo" src="https://sciencedirect.elseviercdn.cn/shared-assets/24/images/elsevier-non-solus-new-grey.svg" alt="Elsevier logo" height="48" width="54">
    <a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253523004748">Information Fusion
    </a>
</div>
<p style="text-align: center; font-family: 'Times New Roman';">
  </a>
    Abstract:
    The denoising diffusion model has received increasing attention in the field of image generation in recent years, thanks to its powerful generation capability. However, diffusion models should be deeply investigated in the field of multi-source image fusion, such as remote sensing pansharpening and multispectral and hyperspectral image fusion (MHIF). 
    In this paper, we introduce a novel {supervised} diffusion model with two conditional modulation modules, specifically designed for the task of multi-source image fusion. 
    These modules mainly consist of a coarse-grained style modulation (CSM) and a fine-grained wavelet modulation (FWM), which aim to disentangle coarse-grained style information and fine-grained frequency information, respectively, thereby generating competitive fused images. Moreover, some essential strategies for the training of the given diffusion model are well discussed, e.g., the selection of training objectives. 
    The superiority of the proposed method is verified compared with recent state-of-the-art (SOTA) techniques by extensive experiments on two multi-source image fusion benchmarks, i.e., pansharpening and MHIF. In addition, sufficient discussions and ablation studies in the experiments are involved to demonstrate the effectiveness of our approach. 
</a>
</p>

News:
- 2024/4/21: **A new paper about [Neural Shrödinger Bridge Matching for Pansharpening](https://arxiv.org/abs/2404.11416) is released on arxiv**.🤗

- 2023/12/4：**Code RELEASED!**:fire: 

- 2023/11/23: **Code will be released soon!**:fire: 

## Quick Overview

The code in this repo supports Pansharpening, Hyperspectral and multispectral image fusion.

<table><tr>
<td><img src="https://raw.githubusercontent.com/294coder/blog_img_bed/main/img3/202311232300466.png" border=0></td>
<td><img src="https://raw.githubusercontent.com/294coder/blog_img_bed/main/img3/202311232301434.png" border=0></td>
</tr></table>

# Instructions

## Dataset

In this office repo, you can find the Pansharpening dataset of [WV3, GF2, and QB](https://github.com/liangjiandeng/PanCollection).

We follow the [PSRT](https://ieeexplore.ieee.org/document/10044141) to implement Hyperspectral and multispectral image fusion. You can find the data we use in [this repo](https://github.com/shangqideng/PSRT).

Other instructions will come soon!


## Citation

If you find our paper useful, please consider citing the following:

```tex
@article{DDIF,
title = {Diffusion model with disentangled modulations for sharpening multispectral and hyperspectral images},
journal = {Information Fusion},
volume = {104},
pages = {102158},
year = {2024},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2023.102158},
url = {https://www.sciencedirect.com/science/article/pii/S1566253523004748},
author = {Zihan Cao and Shiqi Cao and Liang-Jian Deng and Xiao Wu and Junming Hou and Gemine Vivone},
}
```

