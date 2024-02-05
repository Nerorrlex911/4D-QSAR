from pmapper.pharmacophore import Pharmacophore
from pmapper.customize import load_smarts
from pmapper.utils import load_multi_conf_mol
from .data_utils import appendDataLine
import os

import pandas as pd

smarts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smarts_features', 'smarts_features.txt')
smarts_features = load_smarts(smarts_dir)

def calc_desc_mol(mol, descr_num=[4], smarts_features=smarts_features):
    # descr_num - list of int
    """
    Creates descriptors for a single molecule

    Returns
    -------
    result:dict(dict)
    dict of dicts with descriptors for each conformer (the order is the same as in mol.GetConformers()).
    Each dict has the following structure:
        Keys: signatures sep by "|"; values - counts ;  size of dict may vary

    """
    phs = load_multi_conf_mol(mol,smarts_features=smarts_features)
    result = dict()
    for i,ph in enumerate(phs):
        res = dict()
        for n in descr_num:
            res.update(ph.get_descriptors(ncomb=n))
        result[mol.GetConformer(i).GetId()] = res
    return result

def generate_desc_mapping(desc_signatures,save_path):
    """
    Generates mapping from signatures to indices

    Returns
    -------
    desc_mapping:dict
    Dictionary with signatures as keys and indices as values

    """
    pass

def map_desc(desc_signature,desc_mapping:pd.DataFrame):
    """
    Maps signature to index
    if signature is not in mapping, assigns new index and returns it

    desc_signature:str
    Signature
    desc_mapping:DataFrame
    pd.DataFrame(columns=['desc_signature', 'desc_amount'])
    -------
    Returns
    -------
    desc_index:int
    index of signature

    """

    if desc_mapping['desc_signature'].isin([desc_signature]).any():
        desc_mapping.loc[desc_mapping['desc_signature'] == desc_signature, 'desc_amount'] += 1
        desc_index = desc_mapping.loc[desc_mapping['desc_signature'] == desc_signature].index[0]  # 获取已存在的行的索引
    else:
        desc_mapping = appendDataLine(desc_mapping,{'desc_signature': desc_signature, 'desc_amount': 1})
        desc_index = desc_mapping.index[-1]  # 获取新添加的行的索引
    return desc_index

if __name__ == "__main__":
    testDataFrame = pd.DataFrame(columns=['desc_signature', 'desc_amount'])
    print(type(testDataFrame))
    testDataFrame._append({'desc_signature': 'test', 'desc_amount': 1},ignore_index=True)
    pass
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles('CC(C)Oc1nccc2[nH]nc(-c3cc(C(=O)N4CCOCC4)n(C(C)C)c3)c12')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    print(mol.GetNumAtoms())
    ph = Pharmacophore()
    ph.load_from_smarts(mol=mol,smarts=smarts_features)
    res = dict()
    res.update(ph.get_descriptors(ncomb=4))
    # 将字典转换为DataFrame
    res_data = pd.DataFrame.from_dict(res, orient='index').reset_index()
    # 重命名列
    res_data.columns = ['desc', 'time']
    print(res_data)
    # 保存到本地
    res_data.to_csv('res_data.csv', index=False)
    pass
