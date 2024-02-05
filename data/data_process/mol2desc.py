from rdkit import Chem
from rdkit.Chem import AllChem
from .gen_conf import gen_confs_mol
from .gen_conf import serialize_conf
from .gen_conf import deserialize_mol
from .calc_desc import calc_desc_mol
from .calc_desc import map_desc
from .data_utils import appendDataLine
import pandas as pd
import os
import logging

def mol2desc(smiles_data_path,save_path,nconf=5, energy=100, rms=0.5, seed=0, descr_num=[4]):
    smiles_data = pd.read_csv(smiles_data_path,names=['smiles','mol_id','activity'])
    # 结果为DataFrame，第一列为分子id，第二列为构象id，第三列为构象mol格式的字符串
    result = pd.DataFrame(columns=['mol_id', 'conf_id', 'conf_sig'])
    desc_result = pd.DataFrame(columns=['mol_id', 'conf_id', 'desc_index', 'desc_amount'])
    desc_mapping = pd.DataFrame(columns=['desc_signature', 'desc_amount'])
    # 遍历每个分子
    for i in range(len(smiles_data)):
        # 生成分子
        mol = Chem.MolFromSmiles(smiles_data['smiles'][i])
        # 调用gen_conf.py中的gen_confs_mol函数生成分子构象
        mol = gen_confs_mol(mol=mol, nconf=nconf, energy=energy, rms=rms, seed=seed)
        # 计算描述符
        desc = calc_desc_mol(mol=mol, descr_num=descr_num)
        for conf_id,conf in enumerate(mol.GetConformers()):
            logging.info(f'mol2desc: mol_id: {smiles_data["mol_id"][i]} conf_id: {conf_id}')
            for signature in desc[conf_id]:
                # 调用calc_desc.py中的map_desc函数
                map_desc(signature,desc_mapping)
            result = appendDataLine(result,{'mol_id':smiles_data['mol_id'][i],'conf_id':conf_id,'conf_sig':serialize_conf(mol,conf_id)})
    # 清除出现频率最小5%的行
    threshold = desc_mapping['desc_amount'].quantile(0.05)
    # 选取desc_amount大于或等于threshold的行
    desc_mapping = desc_mapping.loc[desc_mapping['desc_amount'] >= threshold]
    # 重新设置desc_mapping的索引
    desc_mapping.reset_index(drop=True, inplace=True)
    # 遍历每个分子的每个构象并写入desc_result
    for mol_id in result['mol_id'].unique():
        conf_sigs = result[result['mol_id']==mol_id]['conf_sig'].tolist()
        logging.info(f'deserialize_mol: mol_id: {mol_id} conf_sigs:{str(len(conf_sigs))}')
        mol = deserialize_mol(conf_sigs)
        desc = calc_desc_mol(descriptors=desc_mapping, mol=mol, descr_num=descr_num)
        for conf in mol.GetConformers():
            i = conf.GetId()
            for signature in desc[i]:
                if desc_mapping['desc_signature'].isin([signature]).any():
                    desc_result = appendDataLine(desc_result,{'mol_id':mol_id,'conf_id':result['conf_id'][i],'desc_index':map_desc(signature,desc_mapping),'desc_amount':desc[i][signature]})
    # 将result、desc_result、desc_mapping写入csv文件
    result.to_csv(os.path.join(save_path,'result.csv'))
    desc_result.to_csv(os.path.join(save_path,'desc_result.csv'))
    desc_mapping.to_csv(os.path.join(save_path,'desc_mapping.csv'))
    return smiles_data,result,desc_result,desc_mapping
    

if __name__ == "__main__":
    mol = Chem.MolFromSmiles('COC1=C(OC)C=C2CN(CCCCNC(=O)C3=CC=CC=C3)CCC2=C1')
    mol = gen_confs_mol(mol=mol)
    desc = calc_desc_mol(mol=mol, descr_num=[4])
    print(len(desc))