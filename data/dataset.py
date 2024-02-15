import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from data_process.mol2desc import mol_to_desc
import os
import logging
import sys

class MolDataSet(Dataset):
    def __init__(self,smiles_data_path,save_path,nconf=5, energy=100, rms=0.5, seed=42, descr_num=[4]) -> None:
        assert os.path.exists(smiles_data_path),'smiles_data_path not exists'
        assert os.path.exists(save_path),'save_path not exists'
        self.smiles_data_path = smiles_data_path
        self.save_path = save_path
        desc_mapping,molecules = mol_to_desc(smiles_data_path=smiles_data_path,save_path=save_path,nconf=nconf, energy=energy, rms=rms, seed=seed, descr_num=descr_num)
        self.smiles_data = smiles_data
        self.desc_mapping = desc_mapping
        self.molecules = molecules
        nmol = len(smiles_data)
        ndesc = len(desc_mapping.desc_mapping)
        # bags: Nmol*Nconf*Ndesc
        self.bags = torch.from_numpy(np.zeros((nmol, nconf, ndesc)))
        # labels: Nmol
        self.labels = torch.from_numpy(np.zeros(nmol))
        for i,molecule in enumerate(molecules):
            mol = molecule.mol
            self.labels[i] = molecule.activity
            for conf in mol.GetConformers():
                descs = desc_mapping.get_conf_desc(conf)
                for index,amount in descs.items():
                    self.bags[i,conf.GetId(),index] = amount                
    def __len__(self):
        return self.bags.shape[0]
    def __getitem__(self, index):
        return self.bags[index],self.labels[index]
    
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')]  # 添加这一行
    )
    logging.info('---start---')
    smiles_data_path = os.path.join(os.getcwd(),'data','datasets','train.csv')
    save_path = os.path.join(os.getcwd(),'data','descriptors','train')
    smiles_data = pd.read_csv(smiles_data_path,names=['smiles','mol_id','activity'])
    '''
    from data_process.gen_conf import gen_confs_mol
    from data_process.gen_conf import serialize_conf
    from data_process.gen_conf import deserialize_mol
    from data_process.calc_desc import calc_desc_mol
    from data_process.calc_desc import map_desc
    from data_process.data_utils import appendDataLine
    from rdkit import Chem
    nconf=5
    energy=100
    rms=0.5
    seed=42
    descr_num=[4]
    # 遍历每个分子
    for i in range(len(smiles_data)):
        # 生成分子
        mol = Chem.MolFromSmiles(smiles_data['smiles'][i])
        # 调用gen_conf.py中的gen_confs_mol函数生成分子构象
        mol = gen_confs_mol(mol=mol, nconf=nconf, energy=energy, rms=rms, seed=seed)
        for conf_id,conf in enumerate(mol.GetConformers()):
            logging.info(f'mol2desc: mol_id: {smiles_data["mol_id"][i]} conf_id: {conf_id}')
    '''
    dataset = MolDataSet(smiles_data_path=smiles_data_path,save_path=save_path,nconf=5, energy=100, rms=0.5, seed=42, descr_num=[4])
    dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)
    for i,(bags,labels) in enumerate(dataloader):
        print(bags)
        print(labels)
    