from pmapper.pharmacophore import Pharmacophore
from pmapper.customize import load_smarts
from pmapper.utils import load_multi_conf_mol
from .data_utils import appendDataLine
import os
import logging
import pandas as pd

# smarts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smarts_features', 'smarts_features.txt')
# smarts_features = load_smarts(smarts_dir)

import json
import numpy as np
from collections import defaultdict, Counter
# class DescMappingOld:
#     def __init__(self,data=pd.DataFrame({
#             'desc_signature': pd.Series(dtype='str'),
#             'desc_amount': pd.Series(dtype=np.int32)
#         })) -> None:
#         self.desc_mapping = data

#     def merge(self, other):
#         # 转为数字
#         self.desc_mapping['desc_amount'] = self.desc_mapping['desc_amount']
#         other.desc_mapping['desc_amount'] = other.desc_mapping['desc_amount']
#         # 将两个desc_mapping连接起来
#         combined = pd.concat([self.desc_mapping, other.desc_mapping])
#         # 将相同的desc_signature对应的desc_amount加起来
#         self.desc_mapping = combined.groupby('desc_signature', as_index=False).sum()

#     def calc_desc_mol(self,mol, descr_num=[4], smarts_features=smarts_features):
#         # descr_num - list of int
#         """
#         Creates descriptors for a single molecule

#         Returns
#         -------
#         result:dict(dict)
#         dict of dicts with descriptors for each conformer (the order is the same as in mol.GetConformers()).
#         Each dict has the following structure:
#             Keys: signatures sep by "|"; values - counts ;  size of dict may vary

#         """
#         logging.info(f'calc_desc_mol: mol_id: {mol.GetProp("_Name")}')
#         phs = load_multi_conf_mol(mol,smarts_features=smarts_features)
#         result = dict()
#         for i,ph in enumerate(phs):
#             res = dict()
#             for n in descr_num:
#                 desc_sig = ph.get_descriptors(ncomb=n)
#                 res.update(desc_sig)
#             self.load_desc(res)
#             mol.GetConformer(i).SetProp("Descriptors", json.dumps(res))
#             result[mol.GetConformer(i).GetId()] = res
#         mol.SetProp("Descriptors", json.dumps(result))
#         return result
#     def load_desc(self,res:dict):
#         for desc_signature,desc_amount in res.items():
#             if self.desc_mapping['desc_signature'].isin([desc_signature]).any():
#                 self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature, 'desc_amount'] += desc_amount
#                 desc_index = self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature].index[0]
#             else:
#                 self.desc_mapping = appendDataLine(self.desc_mapping,{desc_signature: desc_amount})
#                 desc_index = self.desc_mapping.index[-1]  
#     def map_desc(self,desc_signature):
#         if self.desc_mapping['desc_signature'].isin([desc_signature]).any():
#             desc_index = self.desc_mapping.loc[self.desc_mapping['desc_signature'] == desc_signature].index[0]
#         else:
#             desc_index = -1
#         return desc_index
#     def remove_desc(self):
#         # 清除出现频率最小5%的行
#         threshold = self.desc_mapping['desc_amount'].quantile(0.05)
    
#         #缓存清除的数据
#         self.removed = self.desc_mapping.loc[self.desc_mapping['desc_amount'] < threshold]
#         # 选取desc_amount大于或等于threshold的行
#         self.desc_mapping = self.desc_mapping.loc[self.desc_mapping['desc_amount'] >= threshold]
#         logging.info(
#             f'''
#             remove_desc: 
#             threshold: {str(threshold)}
#             removed: {str(len(self.removed))}
#             desc_mapping_len: {str(len(self.desc_mapping))}
#             '''
#         )
#         # 重新设置desc_mapping的索引
#         self.desc_mapping.reset_index(drop=True, inplace=True)
#         return self.removed
#     def load_conf_desc(self, descs):

#         desc_ids = dict()
#         keys_to_remove = []

#         # 将descs中的desc_signature转换为desc_index，并记录不存在的desc_signature
#         for k, v in descs.items():
#             desc_index = self.map_desc(k)
#             if desc_index == -1:
#                 keys_to_remove.append(k)
#             else:
#                 desc_ids[int(desc_index)] = int(v)

#         # 删除不存在的desc_signature
#         for k in keys_to_remove:
#             descs.pop(k)

    
#         return desc_ids
#     def get_conf_desc(self, conf):
#         if conf.HasProp("Descriptors_index"):
#             return json.loads(conf.GetProp("Descriptors_index"))
#         else:
#             return dict()

class DescMapping:
    '''
    desc_amount: Counter(str:desc_signature,int:desc_amount)
    desc_mapping: dict(str:desc_signature,int:desc_id)
    '''
    def __init__(self,desc_amount=Counter(),desc_mapping=dict()) -> None:
        self.desc_amount = desc_amount
        self.desc_mapping = desc_mapping

    def merge(self, other):
        self.desc_amount.update(other.desc_amount)
        return self


    def calc_desc_mol(self,molecule, smarts_features, descr_num=[4]):
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
        logging.info(f'calc_desc_mol: mol_id: {molecule.mol_id}')
        mol = molecule.mol
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
        molecule.mol = mol
        molecule.desc_result = result
        return molecule
    def load_desc(self,res:dict):
        self.desc_amount.update(res)
    def index_desc(self):
        for i,desc_signature in enumerate(self.desc_amount.keys()):
            logging.info(f'index_desc: desc_signature: {desc_signature} will be indexed as {i}')
            self.desc_mapping[desc_signature] = i
    def map_desc(self,desc_signature):
       return self.desc_mapping.get(desc_signature,-1)
    def remove_desc(self):
        # 清除出现频率最小5%的行
        threshold = self.desc_amount.most_common()[-int(len(self.desc_amount) * 0.05)-1][1]
        #缓存清除的数据
        self.removed = dict()
        for desc_signature,desc_amount in self.desc_amount.items():
            if desc_amount < threshold:
                logging.info(f'remove_desc: desc_signature: {desc_signature} desc_amount: {desc_amount}')
                self.removed[desc_signature] = self.desc_amount[desc_signature]
        for desc_signature in self.removed.keys():
            self.desc_amount.pop(desc_signature)
        logging.info(
            f'''
            remove_desc: 
            threshold: {str(threshold)}
            removed: {str(len(self.removed))}
            '''
        )
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
        if conf.HasProp("Descriptors_index"):
            return json.loads(conf.GetProp("Descriptors_index"))
        else:
            return dict()
    def read(self,path):
        if os.path.exists(os.path.join(path, 'desc_mapping.txt')) & os.path.exists(os.path.join(path, 'desc_amount.txt')):           
            with open(os.path.join(path,'desc_amount.txt'),'r') as f:
                self.desc_amount = Counter(json.loads(f.read()))
            with open(os.path.join(path,'desc_mapping.txt'),'r') as f:
                self.desc_mapping = json.loads(f.read())
            return True
        else:
            return False
    def save(self,path):
        with open(os.path.join(path,'desc_amount.txt'),'w') as f:
            f.write(json.dumps(self.desc_amount))
        with open(os.path.join(path,'desc_mapping.txt'),'w') as f:
            f.write(json.dumps(self.desc_mapping))




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
    
