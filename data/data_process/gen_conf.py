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
                    remove_ids.append(i)
                    break
            rms_list = [item for item in rms_list if item[0] != i and item[1] != i]

    for cid in set(remove_ids):
        mol.RemoveConformer(cid)


def gen_confs(mol, mol_name, nconf, energy, rms, seed, act, mol_id):
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=700, randomSeed=seed)

    for cid in cids:
        try:
            #AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        except:
            continue
    remove_confs(mol, energy, rms)
    return mol_name, mol, act, mol_id

if __name__ == "__main__":
    pass