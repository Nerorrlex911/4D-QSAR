from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from data.data_process.gen_conf import gen_confs_mol
if __name__ == "__main__":
    mol = Chem.MolFromSmiles('CC(C)Oc1nccc2[nH]nc(-c3cc(C(=O)N4CCOCC4)n(C(C)C)c3)c12')
    mol = PropertyMol(mol)
    mol = gen_confs_mol(mol=mol,nconf=5)
    mol.GetConformers()[0].SetProp("Descriptor_index","111")
    mol.SetProp("MyProperty", "MyValue")
    with Chem.SDWriter('result.sdf') as w:
        w.write(mol)
    supplier = Chem.SDMolSupplier('result.sdf')
    for mol in supplier:
        print(mol.GetProp("MyProperty"))
        print(mol.GetConformers()[0].GetProp("Descriptor_index"))  