import numpy as np
import torch.utils.data as data

def non_wavelet_ref_loop(dl: data.DataLoader):
    for i, (pan, lms, gt) in enumerate(dl):
        yield i, pan, lms, gt
        
def wavelet_ref_loop(dl: data.DataLoader):
    for i, (pan, lms, gt, wavelets) in enumerate(dl):
        yield i, pan, lms, wavelets

def non_wavelet_unref_loop(dl: data.DataLoader):
    for i, (pan, lms) in enumerate(dl):
        yield i, pan, lms
        
def wavelet_unref_loop(dl: data.DataLoader):
    for i, (pan, lms, wavelets) in enumerate(dl):
        yield i, pan, lms, wavelets