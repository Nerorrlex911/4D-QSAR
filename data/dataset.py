import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from data.data_process.mol2desc import mol_to_desc
import os
import logging
import sys

class MolDataSet(Dataset):
    def __init__(self,smiles_data_path,save_path,nconf=5, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=10,new=False) -> None:
        assert os.path.exists(smiles_data_path),'smiles_data_path not exists'
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        self.smiles_data_path = smiles_data_path
        self.save_path = save_path
        
        desc_mapping,molecules = mol_to_desc(smiles_data_path=smiles_data_path,save_path=save_path,nconf=nconf, energy=energy, rms=rms, seed=seed, descr_num=descr_num,ncpu=ncpu)
    
        self.desc_mapping = desc_mapping
        self.molecules = molecules
        nmol = len(molecules)
        ndesc = len(desc_mapping.desc_mapping)
        # bags: Nmol*Nconf*Ndesc 训练数据
        self.bags = torch.from_numpy(np.zeros((nmol, nconf, ndesc),dtype=np.float32))
        # mask: Nmol*Nconf*1 标记哪些构象是有效的，在训练过程中去除噪点
        self.mask = torch.from_numpy(np.ones((nmol, nconf,1),dtype=np.float32))
        # labels: Nmol
        self.labels = torch.from_numpy(np.zeros(nmol,dtype=np.float32))
        for i,molecule in enumerate(molecules):
            mol = molecule.mol
            self.labels[i] = float(molecule.activity)
            self.mask[i][mol.GetNumConformers():] = 0.0
            for conf in mol.GetConformers():
                descs = desc_mapping.get_conf_desc(conf)
                for index,amount in descs.items():
                    self.bags[i,int(conf.GetId()),int(index)] = float(amount)               
    def __len__(self):
        return self.bags.shape[0]
    def __getitem__(self, index):
        return (self.bags[index],self.mask[index]),self.labels[index]
    
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')] 
    )
    logging.info('---start---')
    
    data_path = os.path.join(os.getcwd(),'data','datasets','train.csv')
    save_path = os.path.join(os.getcwd(),'data','descriptors','train')
    # 加载数据集
    generator = torch.Generator().manual_seed(42)
    dataset = MolDataSet(data_path,save_path)
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False)
    for i,((bags,mask),labels) in enumerate(dataloader):
        print(bags)
        print(mask)
        print(labels)
    