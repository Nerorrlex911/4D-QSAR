
import os
from miqsar.utils import calc_3d_pmapper
import os
import pickle
import numpy as np
import pandas as pd

def load_svm_data(fname):
    
    def str_to_vec(dsc_str, dsc_num): 

        tmp = {}
        for i in dsc_str.split(' '):
            tmp[int(i.split(':')[0])] = int(i.split(':')[1])
        #
        tmp_sorted = {}
        for i in range(dsc_num):
            tmp_sorted[i] = tmp.get(i, 0)
        vec = list(tmp_sorted.values())

        return vec
    
    #
    with open(fname) as f:
        dsc_tmp = [i.strip() for i in f.readlines()]

    with open(fname.replace('txt', 'rownames')) as f:
        mol_names = [i.strip() for i in f.readlines()]
    #
    labels_tmp = [float(i.split(':')[1]) for i in mol_names]
    idx_tmp = [i.split(':')[0] for i in mol_names]
    dsc_num = max([max([int(j.split(':')[0]) for j in i.strip().split(' ')]) for i in dsc_tmp])
    #
    bags, labels, idx = [], [], []
    for mol_idx in list(np.unique(idx_tmp)):
        bag, labels_, idx_ = [], [], []
        for dsc_str, label, i in zip(dsc_tmp, labels_tmp, idx_tmp):
            if i == mol_idx:
                bag.append(str_to_vec(dsc_str, dsc_num))
                labels_.append(label)
                idx_.append(i)
                
        bags.append(np.array(bag).astype('uint8'))
        labels.append(labels_[0])
        idx.append(idx_[0])

    return np.array(bags), np.array(labels), np.array(idx)




# 主函数
if __name__ == "__main__":
    bags = np.load('bags.npy')
    print(
        'bags.npy loaded',
        'bags.shape:', bags.shape,
        'bags[0].shape:', bags[0].shape,
        )
    exit()
    #Choose dataset to be modeled and create a folder where the descriptors will be stored
    dataset = "train"
    nconfs_list = [1, 5] #number of conformations to generate; calculation is time consuming, so here we set 5, for real tasks set 25..100
    ncpu = 6 # set number of CPU cores 

    dataset_file = os.path.join('data','datasets', dataset+'.smi')
    descriptors_folder = os.path.join('data','descriptors')
    # os.mkdir(descriptors_folder)

    out_fname = calc_3d_pmapper(input_fname=dataset_file, nconfs_list=nconfs_list, energy=100,  descr_num=[4],
                            path=descriptors_folder, ncpu=ncpu)
    
    # split data into a training and test set
    dsc_fname = os.path.join(descriptors_folder, f'PhFprPmapper_conf-{dataset}_5.txt') # descriptors file
    bags, labels, idx = load_svm_data(dsc_fname)
    print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')
    np.save('bags.npy', bags)
    