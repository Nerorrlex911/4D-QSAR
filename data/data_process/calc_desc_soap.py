from typing import Iterable, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem import Mol
from .gen_conf import gen_confs_mol, serialize_conf, deserialize_mol
import json
import numpy as np
from ase import Atoms
from data.data_process.Molecule import Molecule
from dscribe.descriptors import SOAP
import scipy
import multiprocessing
def get_atom_types(molecules:Iterable[Molecule]):
    atom_types = set()
    for molecule in molecules:
        mol = molecule.mol
        atom_types=atom_types.union([atom.GetSymbol() for atom in mol.GetAtoms()])
    return atom_types
def conf_to_Atoms(rdkit_molecule:Mol,conf_id:int,atom_types):
    # 提取坐标
    conf = rdkit_molecule.GetConformer(conf_id)
    coords = np.array([conf.GetAtomPosition(i) for i in range(rdkit_molecule.GetNumAtoms())])
    ase_atoms = Atoms(symbols=atom_types, positions=coords)
    return ase_atoms
def process_desc(args:Tuple[Molecule,SOAP,SOAP]):
    molecule,small_soap,large_soap = args
    mol = molecule.mol
    #提取原子类型
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for i,conf in enumerate(mol.GetConformers()):
        conf_id = conf.GetId()
        ase_atoms = conf_to_Atoms(mol,conf_id,atom_types)
        # 生成SOAP描述符
        small_soap_desc = small_soap.create(ase_atoms)
        large_soap_desc = large_soap.create(ase_atoms)
        soap_desc = np.hstack([small_soap_desc, large_soap_desc])
        # 将描述符存入molecule对象
        molecule.desc_result[conf_id] = soap_desc
    return molecule
def calc_desc_soap(molecules:Iterable[Molecule],ncpu:int):
    species = get_atom_types(molecules)
    # Setting up the SOAP descriptor
    rcut_small = 3.0
    sigma_small = 0.2
    rcut_large = 6.0
    sigma_large = 0.4

    small_soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut_small,
        n_max=12,
        l_max=8,
        sigma = sigma_small,
        sparse=False,
        average="inner"
    )

    large_soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut_large,
        n_max=12,
        l_max=8,
        sigma = sigma_large,
        sparse=False,
        average="inner"
    )

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule,small_soap,large_soap) for molecule in molecules]
        molecules = pool.map(process_desc,args)
    return molecules

def calc_desc_soap_flat(molecules:Iterable[Molecule],ncpu:int):
    species = get_atom_types(molecules)
    # Setting up the SOAP descriptor
    rcut_small = 3.0
    sigma_small = 0.2
    rcut_large = 6.0
    sigma_large = 0.4

    small_soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut_small,
        n_max=12,
        l_max=8,
        sigma = sigma_small,
        sparse=False,
        compression={"mode": "mu2"}
    )

    large_soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut_large,
        n_max=12,
        l_max=8,
        sigma = sigma_large,
        sparse=False,
        compression={"mode": "mu2"}
    )

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule,small_soap,large_soap) for molecule in molecules]
        molecules = pool.map(process_desc,args)
    # 获取所有分子中最大的原子数量
    max_atom_num = max([molecule.desc_result[0].shape[0] for molecule in molecules])
    print(f'max_atom_num:{max_atom_num}')
    # 将所有分子的描述符扩展到相同的维度再展平
    for molecule in molecules:
        for conf_id,desc in molecule.desc_result.items():
            molecule.desc_result[conf_id] = np.pad(desc,((0,max_atom_num-desc.shape[0]),(0,0))).flatten()
    return molecules