

import pandas as pd


import multiprocessing
import os
import pickle

from data.data_process.Molecule import Molecule


def process_conf(args):
    row, nconf, energy, rms, seed = args
    molecule = Molecule(row['smiles'], row['mol_id'], row['activity'])
    molecule.gen_confs(nconf=nconf, energy=energy, rms=rms, seed=seed)
    return molecule


def mol_to_conf(smiles_data_path, save_path, nconf=2, energy=100, rms=0.5, seed=42,ncpu=10,new=False):
    molecules = []
    smiles_data = pd.read_csv(smiles_data_path, names=['smiles', 'mol_id', 'activity'])
    #如果已经存在构象结果则直接加载
    if (not new) & os.path.exists(os.path.join(save_path, 'conf_result.pkl')):
        with open(os.path.join(save_path, 'conf_result.pkl'), 'rb') as f:
            molecules = pickle.load(f)
    else:
        with multiprocessing.Pool(ncpu) as pool:
            args = [(row, nconf, energy, rms, seed) for _, row in smiles_data.iterrows()]
            molecules = pool.map(process_conf, args)
        with open(os.path.join(save_path, 'conf_result.pkl'), 'wb') as f:
            pickle.dump(molecules, f)
    return molecules