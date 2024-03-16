#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import h5py
from tabulate import tabulate
from tqdm import tqdm, trange

from utils import AnalysisPanAcc, find_data_path, h5py_to_dict

def metric_dicts_ave_and_std(metrics: list[dict]):
    ave = {}
    std = {}
    
    keys = metrics[0].keys()
    
    for k in keys:
        ave[k] = np.mean([m[k] for m in metrics])
        std[k] = np.std([m[k] for m in metrics])
        
    return ave, std

def to_tabulate(ave: dict, std: dict):
    table = []
    for k in ave.keys():
        table.append([k, ave[k], std[k]])
        
    return tabulate(table, headers=["Metric", "Average", "Standard Deviation"], tablefmt="rounded_grid")


def norm_to_0_1(*args, norm_const=2047):
    return [a / norm_const for a in args] if len(args) > 1 else args[0] / norm_const


#%%


path = '/home/czh/exps/fcformer-bk/visualized_img/data_panMamba_wv3_unref_p64.mat'
full_res = True if 'unref' else False
dataset_type = 'wv3'
ratio = 4
const = {'wv3': 2047, 'gf': 1023, 'qb': 1023, 'wv2': 2047,
         'cave': 1, 'harvard': 1, 'gf5': 1}.get(dataset_type.split('_')[0], 1)
dataset_path = find_data_path(dataset_type, full_res)

dataset = h5py.File(dataset_path, 'r')

if dataset_type[:4] == "cave" or dataset_type[:7] == "harvard":
    keys = ["LRHSI", "HSI_up", "RGB", "GT"]
else:
    keys = None
dataset = h5py_to_dict(dataset, keys)
analysis = AnalysisPanAcc(ref=not full_res, ratio=ratio, sensor=dataset_type)

path = Path(path)
if path.is_dir():
    file_lst = list(path.glob('*.mat'))
    metrics = []
    for i, path in tqdm(enumerate(file_lst)):
        ms, lms, pan = dataset["ms"][i:i+1], dataset["lms"][i:i+1], dataset["pan"][i:i+1]
        sr = loadmat(path)['sr']
        ms, lms, pan, sr = norm_to_0_1(ms, lms, pan, sr, norm_const=const)
        if full_res: 
            analysis(sr, ms, lms, pan)
        else:
            gt = dataset["gt"][i:i+1]
            gt = norm_to_0_1(gt, norm_const=const)
            analysis(sr, gt)
        metrics.append(analysis.acc_ave)
        analysis.clear_history()
        
    ave, std = metric_dicts_ave_and_std(metrics)
    table = to_tabulate(ave, std)
else:
    files = loadmat(path)['sr']
    metrics = []
    for i in trange(len(files)):
        ms, lms, pan = dataset["ms"][i:i+1], dataset["lms"][i:i+1], dataset["pan"][i:i+1]
        sr = files[i:i+1]
        ms, lms, pan, sr = norm_to_0_1(ms, lms, pan, sr, norm_const=const)
        if full_res: 
            analysis(sr, ms, lms, pan)
        else:
            gt = dataset["gt"][i:i+1]
            gt = norm_to_0_1(gt, norm_const=const)
            analysis(sr, gt)
        metrics.append(analysis.acc_ave)
        analysis.clear_history()
        
    ave, std = metric_dicts_ave_and_std(metrics)
    table = to_tabulate(ave, std)
    
print(table)
    
    
    




    

