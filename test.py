from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
if __name__ == "__main__":
    mol = Chem.MolFromSmiles('CC(C)Oc1nccc2[nH]nc(-c3cc(C(=O)N4CCOCC4)n(C(C)C)c3)c12')
    mol = PropertyMol(mol)
    mol.SetProp("MyProperty", "MyValue")
    with Chem.SDWriter('result.sdf') as w:
        w.write(mol)
    supplier = Chem.SDMolSupplier('result.sdf')
    for mol in supplier:
        print(mol.GetProp("MyProperty"))  # 输出: MyValue