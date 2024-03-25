import h5py
import numpy as np
import os
import os.path as osp
import glob
import scipy.io as io
from tqdm import tqdm


# file = r'/Data2/ZiHanCao/exps/panformer/visualized_img/data_dcformermwsa_new_cave_x8_ref.mat'


def process_mat_to_single_mat(file):
    name = file.split('/')[-1].strip('.mat')
    path = f'/Data2/ZiHanCao/exps/panformer/visualized_img/{name}'
    print(path)
    save_prefix = 'output_mulExm_'
    if not osp.exists(path):
        os.mkdir(path)
        print(f'make dir {name}')
    else:
        print('already process this dir')
        return
        
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
            
    print('---------------'*10)
            
files = glob.glob('/Data2/ZiHanCao/exps/panformer/visualized_img/*.mat')

print('found files: ', files)


for file in files:
    process_mat_to_single_mat(file)
    

        