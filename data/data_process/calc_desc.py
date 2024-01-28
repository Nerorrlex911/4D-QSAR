from pmapper.pharmacophore import Pharmacophore
from pmapper.customize import load_smarts
from pmapper.utils import load_multi_conf_mol
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
    result:list(dict)
    List of dicts with descriptors for each conformer (the order is the same as in mol.GetConformers()).
    Each dict has the following structure:
        Keys: signatures sep by "|"; values - counts ;  size of dict may vary

    """
    phs = load_multi_conf_mol(mol,smarts_features=smarts_features)
    result = []
    for ph in phs:
        res = dict()
        for n in descr_num:
            res.update(ph.get_descriptors(ncomb=n))
        result.append(res)
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
        desc_mapping = desc_mapping.append({'desc_signature': desc_signature, 'desc_amount': 1})
        desc_index = desc_mapping.index[-1]  # 获取新添加的行的索引
    return desc_index

if __name__ == "__main__":
    test = dict()
    print(test)
    pass
