from pmapper.pharmacophore import Pharmacophore
from pmapper.customize import load_smarts
from pmapper.utils import load_multi_conf_mol
from .data_utils import appendDataLine
import os
import logging
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

import json
import numpy as np
class DescMapping:
    def __init__(self,data=pd.DataFrame({
            'desc_signature': pd.Series(dtype='str'),
            'desc_amount': pd.Series(dtype=np.int64)
        })) -> None:
        self.desc_mapping = data

    def merge(self, other):
        # 转为数字
        self.desc_mapping['desc_amount'] = self.desc_mapping['desc_amount'].astype(int)
        other.desc_mapping['desc_amount'] = other.desc_mapping['desc_amount'].astype(int)
        # 将两个desc_mapping连接起来
        combined = pd.concat([self.desc_mapping, other.desc_mapping])
        # 将相同的desc_signature对应的desc_amount加起来
        self.desc_mapping = combined.groupby('desc_signature', as_index=False).sum()

    def calc_desc_mol(self,mol, descr_num=[4], smarts_features=smarts_features):
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
                desc_sig = ph.get_descriptors(ncomb=n)
                res.update(desc_sig)
            self.load_desc(res)
            mol.GetConformer(i).SetProp("Descriptors", json.dumps(res))
            result[mol.GetConformer(i).GetId()] = res
        mol.SetProp("Descriptors", json.dumps(result))
        return result
    def load_desc(self,res:dict):
        for desc_signature,desc_amount in res.items():
            if self.desc_mapping['desc_signature'].isin([desc_signature]).any():
                self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature, 'desc_amount'] += desc_amount
                desc_index = self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature].index[0]
            else:
                self.desc_mapping = appendDataLine(self.desc_mapping,{desc_signature: desc_amount})
                desc_index = self.desc_mapping.index[-1]  # 获取新添加的行的索引
    def map_desc(self,desc_signature):
        if self.desc_mapping['desc_signature'].isin([desc_signature]).any():
            desc_index = self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature].index[0]
        else:
            desc_index = -1
        return desc_index
    def remove_desc(self):
        # 清除出现频率最小5%的行
        threshold = self.desc_mapping['desc_amount'].quantile(0.05)
    
        #缓存清除的数据
        self.removed = self.desc_mapping.loc[self.desc_mapping['desc_amount'] < threshold]
        # 选取desc_amount大于或等于threshold的行
        self.desc_mapping = self.desc_mapping.loc[self.desc_mapping['desc_amount'] >= threshold]
        logging.info(
            f'''
            remove_desc: 
            threshold: {str(threshold)}
            removed: {str(len(self.removed))}
            desc_mapping_len: {str(len(self.desc_mapping))}
            '''
        )
        # 重新设置desc_mapping的索引
        self.desc_mapping.reset_index(drop=True, inplace=True)
        return self.removed
    def load_conf_desc(self, descs):

        desc_ids = dict()
        keys_to_remove = []

        # 将descs中的desc_signature转换为desc_index，并记录不存在的desc_signature
        for k, v in descs.items():
            desc_index = self.map_desc(k)
            if desc_index == -1:
                keys_to_remove.append(k)
            else:
                desc_ids[int(desc_index)] = int(v)

        # 删除不存在的desc_signature
        for k in keys_to_remove:
            descs.pop(k)

    
        return desc_ids
    def get_conf_desc(self, conf):
        return json.loads(conf.GetProp("Descriptors_index"))






if __name__ == "__main__":
    # 导入pandas库
    import pandas as pd

    # 创建两个数据框
    testDataFrame = pd.DataFrame({
        'desc_signature': ['a', 'b', 'c', 'd', 'e'],
        'desc_amount': [1, 2, 3, 4, 5]
    })
    testDataFrame2 = pd.DataFrame({
        'desc_signature': ['f', 'g', 'c', 'd', 'e'],
        'desc_amount': [1, 2, 3, 4, 5]
    })

    # 将两个数据框连接起来
    combined = pd.concat([testDataFrame, testDataFrame2])

    # 对desc_signature相同的行进行求和
    result = combined.groupby('desc_signature', as_index=False).sum()

    print(result)
    
