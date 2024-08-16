from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from pmapper.customize import load_smarts
from data.data_process.mol_to_conf import mol_to_conf
from .gen_conf import gen_confs_mol
from .data_utils import divide_list
from .calc_desc import DescMapping
import os
import logging
import pickle
import multiprocessing

def process_desc(args):
    molecule,descr_num,smarts_features = args
    desc_mapping = DescMapping()
    molecule = desc_mapping.calc_desc_mol(molecule=molecule,smarts_features=smarts_features, descr_num=descr_num)
    return molecule, desc_mapping

def map_desc(args):
    molecule, desc_mapping = args
    logging.info(f'mol_to_desc> map_desc: mol_id: {molecule.mol_id}')
    desc_result = molecule.desc_result
    new_desc_result = dict()
    for conf_id, descs in desc_result.items():
        new_desc_result[conf_id] = desc_mapping.load_conf_desc(descs)
    molecule.desc_result = new_desc_result
    return molecule

def merge_desc(args):
    desc_mapping_result, sub_desc_mappings = args
    logging.info(f'mol_to_desc> merge_desc: {sub_desc_mappings.__len__()}')
    for sub_desc_mapping in sub_desc_mappings:
        desc_mapping_result = desc_mapping_result.merge(sub_desc_mapping)
    return desc_mapping_result

def mol_to_desc(smiles_data_path, save_path, nconf=2, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=10,new=False):
    smarts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smarts_features', 'smarts_features.txt')
    smarts_features = load_smarts(smarts_dir)
    molecules = []
    desc_mapping = DescMapping()
    if (not new) & desc_mapping.read(save_path) & os.path.exists(os.path.join(save_path, 'molecules_result.pkl')):
        with open(os.path.join(save_path, 'molecules_result.pkl'), 'rb') as f:
            molecules = pickle.load(f)
        return desc_mapping, molecules
    else:
        molecules = mol_to_conf(smiles_data_path=smiles_data_path, save_path=save_path, nconf=nconf, energy=energy, rms=rms, seed=seed, ncpu=ncpu,new=new)

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule,descr_num,smarts_features) for molecule in molecules]
        results = pool.map(process_desc,args)
        molecules, desc_mappings = zip(*results)

    desc_mapping_results = [DescMapping() for _ in range(ncpu)]

    with multiprocessing.Pool(ncpu) as pool:
        args = [(desc_mapping_results[index], sub_desc_mappings) for index,sub_desc_mappings in enumerate(divide_list(desc_mappings,ncpu))]
        desc_mappings = pool.map(merge_desc, args)

    for i,dm in enumerate(desc_mappings):
        desc_mapping.merge(dm)

    confs_amount = 0
    for i,molecule in enumerate(molecules):
        confs_amount = confs_amount+len(molecule.mol.GetConformers())
    threshold = confs_amount*0.01

    desc_mapping.remove_desc(threshold=threshold)

    desc_mapping.index_desc()
    
    desc_mapping.save(save_path)

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule, desc_mapping) for molecule in molecules]
        molecules = pool.map(map_desc, args)

    with open(os.path.join(save_path, 'molecules_result.pkl'), 'wb') as f:
        pickle.dump(molecules, f)

    return desc_mapping, molecules
