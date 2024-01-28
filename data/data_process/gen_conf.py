import os, time
import sys
import gzip
import argparse
import pickle
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from openbabel import pybel


def remove_confs(mol, energy, rms):
    e = []
    for conf in mol.GetConformers():
        #ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())

        if ff is None:
            print(Chem.MolToSmiles(mol))
            return
        e.append((conf.GetId(), ff.CalcEnergy()))
    e = sorted(e, key=lambda x: x[1])

    if not e:
        return

    kept_ids = [e[0][0]]
    remove_ids = []

    for item in e[1:]:
        if item[1] - e[0][1] <= energy:
            kept_ids.append(item[0])
        else:
            remove_ids.append(item[0])

    if rms is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(kept_ids, 2)]
        while any(item[2] < rms for item in rms_list):
            for item in rms_list:
                if item[2] < rms:
                    i = item[1]
                    print('removing', i)
                    remove_ids.append(i)
                    break
            rms_list = [item for item in rms_list if item[0] != i and item[1] != i]
    print(remove_ids)
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
# 重新排列分子的构象，保证构象加入的顺序与构象id从小到大的顺序一致
def reorder_confs(mol):
    """
    Reorder conformers in a molecule
    重新排列分子的构象，改变构象id以保证构象加入的顺序与构象id从小到大的顺序一致

    Returns
    -------
    mol

    """
    confs = mol.GetConformers()
    confs = sorted(confs, key=lambda x: x.GetId())
    for i, conf in enumerate(confs):
        conf.SetId(i)
    return mol
# 生成单个分子的构象
def gen_confs_mol(mol, nconf=5, energy=100, rms=0.5, seed=0):
    """
    Generates conformations for a single molecule

    Returns
    -------
    mol:Mol with conformations

    """
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=700, randomSeed=seed)

    for cid in cids:
        try:
            #AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        except:
            continue
    remove_confs(mol, energy, rms)
    return reorder_confs(mol)

#使用Mol格式序列化和反序列化分子构象
#序列化单个构象
def serialize_conf(mol,id:int):
    """
    Serialize molecule with single conformation to string

    Returns
    -------
    string

    """
    return Chem.MolToMolBlock(mol,confId=id)
#反序列化单个构象
def deserialize_conf(s):
    """
    Deserialize molecule with single conformation from string

    Returns
    -------
    mol

    """
    return Chem.MolFromMolBlock(s)
# 序列化多个构象
def serialize_mol(mol):
    """
    Serialize molecule with multiple conformations to string

    Returns
    -------
    list(string)

    """
    return [Chem.rdmolfiles.MolToMolBlock(mol, confId=cid) for cid in mol.GetConformerIds()]
# 反序列化多个构象
def deserialize_mol(s):
    """
    Deserialize molecule with multiple conformations from string

    Returns
    -------
    mol

    """
    mol = Chem.MolFromMolBlock(s)
    for conf in mol.GetConformers():
        conf.SetId(int(conf.GetProp('_Name')))
    return mol

if __name__ == "__main__":
    mol = Chem.MolFromSmiles('CC(C)Oc1nccc2[nH]nc(-c3cc(C(=O)N4CCOCC4)n(C(C)C)c3)c12')
    mol = gen_confs_mol(mol=mol,nconf=5)
    print(mol.GetNumConformers())
    for conf in mol.GetConformers():
        print(conf.GetId())