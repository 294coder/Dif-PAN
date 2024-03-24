import os

from torch.utils.cpp_extension import load

cwd = os.path.dirname(__file__)

T_MAX = 8192
wkv_cuda = load(name="wkv", sources=[f"{cwd}/wkv_op.cpp", f"{cwd}/wkv_cuda.cu"], build_directory=cwd,
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

__all__ = ['wkv_cuda']