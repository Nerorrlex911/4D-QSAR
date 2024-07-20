import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from data.data_process.mol2desc import mol_to_desc
import os
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def scale_data(x_train, x_val, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_val):
        x_val_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)
    return np.array(x_train_scaled), np.array(x_val_scaled), np.array(x_test_scaled)

class MolDataSet(Dataset):
    def __init__(self,bags,mask,labels) -> None:
        # bags: Nmol*Nconf*Ndesc 训练数据
        self.bags = torch.from_numpy(bags)
        # mask: Nmol*Nconf*1 标记哪些构象是有效的，在训练过程中去除噪点
        self.mask = torch.from_numpy(mask)
        # labels: Nmol
        self.labels = torch.from_numpy(labels)   
                   
    def __len__(self):
        return self.bags.shape[0]
    def __getitem__(self, index):
        return (self.bags[index],self.mask[index]),self.labels[index]

class MolData:
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
        self.bags = np.zeros((nmol, nconf, ndesc),dtype=np.float32)
        # mask: Nmol*Nconf*1 标记哪些构象是有效的，在训练过程中去除噪点
        self.mask = np.ones((nmol, nconf,1),dtype=np.float32)
        # labels: Nmol
        self.labels = np.zeros(nmol,dtype=np.float32)
        for i,molecule in enumerate(molecules):
            mol = molecule.mol
            self.labels[i] = float(molecule.activity)
            self.mask[i][mol.GetNumConformers():] = 0
            for conf in mol.GetConformers():
                descs = molecule.get_conf_desc(conf.GetId())
                for index,amount in descs.items():
                    self.bags[i,int(conf.GetId()),int(index)] = amount 
    def preprocess(self) -> Tuple[MolDataSet,MolDataSet,MolDataSet]:
        # 首先，将数据集划分为训练集和测试集（70%训练，30%测试）
        x_train, x_test, m_train, m_test, y_train, y_test = train_test_split(self.bags, self.mask, self.labels, test_size=0.3, random_state=42)
        # 接下来，将测试集划分为验证集和测试集（10%验证，20%测试）
        x_val, x_test, m_val, m_test, y_val, y_test = train_test_split(x_test, m_test, y_test, test_size=0.67, random_state=42)
        x_train_scaled, x_val_scaled, x_test_scaled = scale_data(x_train, x_val, x_test)
        return MolDataSet(x_train_scaled,m_train,y_train),MolDataSet(x_val_scaled,m_val,y_val),MolDataSet(x_test_scaled,m_test,y_test)

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
    