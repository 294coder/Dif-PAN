import h5py
import numpy as np
import os
import os.path as osp
import scipy.io as io
from tqdm import tqdm


file = r'/home/czh/exps/fcformer/visualized_img/data_panRWKV_wv3_unref.mat'

name = file.split('/')[-1].strip('.mat')
path = f'/home/czh/exps/fcformer/visualized_img/{name}'
print(path)
save_prefix = 'output_mulExm_'
if not osp.exists(path):
    os.mkdir(path)
    print(f'make dir {name}')
    
mat_file = io.loadmat(file)
print(f'has keys: {mat_file.keys()}')

sr = mat_file.get('sr')
if sr is None:
    print('has no key sr')
else:
    bar = tqdm(range(sr.shape[0]))
    for i in bar:
        save_path = osp.join(path, save_prefix+f'{i}.mat')
        sr_i = np.transpose(sr[i, ...], [1,2,0])
        save_d = {'sr': sr_i}
        io.savemat(save_path, save_d)
        bar.set_description(f'save {i}.mat')
        