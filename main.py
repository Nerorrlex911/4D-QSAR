if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit.Chem.PropertyMol import PropertyMol
    import pickle

# 创建一个Mol对象
    mol = PropertyMol(Chem.MolFromSmiles('CCO'))

# 设置一个属性
    mol.SetProp("MyProperty", "MyValue")

# 将Mol对象序列化为一个pickle字符串
    mol_pickle = pickle.dumps(mol)

# 将pickle字符串转换回Mol对象
    mol = pickle.loads(mol_pickle)

# 获取属性
    print(mol.GetProp("MyProperty"))  # 输出: MyValue