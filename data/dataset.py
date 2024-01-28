from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from data_process.mol2desc import mol2desc

class MolDataSet(Dataset):
    def __init__(self,smiles_data_path,save_path,nconf=5, energy=100, rms=0.5, seed=0, descr_num=[4]) -> None:
        self.smiles_data_path = smiles_data_path
        self.save_path = save_path
        smiles_data,result,desc_result,desc_mapping = mol2desc(smiles_data_path=smiles_data_path,save_path=save_path,nconf=nconf, energy=energy, rms=rms, seed=seed, descr_num=descr_num)
        self.smiles_data = smiles_data
        self.result = result
        self.desc_result = desc_result
        self.desc_mapping = desc_mapping
        nmol = len(smiles_data)
        ndesc = len(desc_mapping)
        # bags: Nmol*Nconf*Ndesc
        self.bags = np.zeros((nmol, nconf, ndesc))
        for mol_id in enumerate(desc_result['mol_id']):
            for conf_id in enumerate(desc_result[desc_result['mol_id']==mol_id,'conf_id']):
                for desc_id in enumerate(desc_result[desc_result['mol_id']==mol_id & desc_result['conf_id']==conf_id,'desc_index']):
                    self.bags[mol_id, conf_id, desc_id] = desc_result[(desc_result['mol_id']==mol_id) & (desc_result['conf_id']==conf_id) & (desc_result['desc_index']==desc_id)]['desc_amount']
        # labels: Nmol
        self.labels = smiles_data['activity'].to_numpy()
