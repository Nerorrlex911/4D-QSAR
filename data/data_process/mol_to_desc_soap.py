from data.data_process.calc_desc_soap import calc_desc_soap
from data.data_process.mol_to_conf import mol_to_conf


import os
import pickle


def mol_to_desc_soap(smiles_data_path, save_path, nconf=2, energy=100, rms=0.5, seed=42, ncpu=10,new=False):
    molecules = []
    if (not new) & os.path.exists(os.path.join(save_path, 'molecules_result_soap.pkl')):
        with open(os.path.join(save_path, 'molecules_result_soap.pkl'), 'rb') as f:
            molecules = pickle.load(f)
        return molecules
    else:
        molecules = mol_to_conf(smiles_data_path=smiles_data_path, save_path=save_path, nconf=nconf, energy=energy, rms=rms, seed=seed, ncpu=ncpu,new=new)
    molecules = calc_desc_soap(molecules,ncpu)
    with open(os.path.join(save_path, 'molecules_result_soap.pkl'), 'wb') as f:
        pickle.dump(molecules, f)
    return molecules