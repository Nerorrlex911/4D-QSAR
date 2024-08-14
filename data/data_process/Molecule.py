from rdkit.Chem.PropertyMol import PropertyMol
from rdkit import Chem
from .gen_conf import gen_confs_mol
class Molecule:
    def __init__(self,smiles_str=None,mol_id=None,activity=None):
        self.desc_result = dict()
        self.smiles_str = smiles_str
        self.mol_id = mol_id
        self.activity = activity
        self.mol = PropertyMol(Chem.MolFromSmiles(smiles_str))
        self.mol.SetProp("_Name", str(mol_id))
        self.mol.SetProp("Activity", str(activity))
    def gen_confs(self,nconf=2, energy=100, rms=0.5, seed=42):
        self.mol = gen_confs_mol(mol=self.mol,nconf=nconf, energy=energy, rms=rms, seed=seed)
    def get_conf_desc(self,conf_id):
        return self.desc_result[conf_id]
    pass