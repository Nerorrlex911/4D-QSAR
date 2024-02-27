from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem.PropertyMol import PropertyMol
from .gen_conf import gen_confs_mol, serialize_conf, deserialize_mol
from .data_utils import appendDataLine
from .calc_desc import DescMapping
import pandas as pd
import os
import logging
import json

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

def mol_to_desc_backup(smiles_data_path,save_path,nconf=5, energy=100, rms=0.5, seed=42, descr_num=[4]):
    smiles_data = pd.read_csv(smiles_data_path,names=['smiles','mol_id','activity'])
    desc_mapping = DescMapping()
    molecules = []
    for i in range(len(smiles_data)):
        molecule = Molecule(smiles_data['smiles'][i],smiles_data['mol_id'][i],smiles_data['activity'][i])
        molecule.gen_confs(nconf=nconf, energy=energy, rms=rms, seed=seed)
        desc_mapping.calc_desc_mol(mol=molecule.mol, descr_num=descr_num)
        molecules.append(molecule)
    desc_mapping.remove_desc()
    #遍历每个分子的每个构象
    for molecule in molecules:
        for conf in molecule.mol.GetConformers():
            conf_desc = desc_mapping.load_conf_desc(conf=conf)
    #保存
    desc_mapping.desc_mapping.to_csv(os.path.join(save_path,'desc_mapping.csv'))
    with Chem.SDWriter(os.path.join(save_path,'result.sdf')) as w:
        for m in molecules:
            w.write(m.mol)
    return desc_mapping,molecules

import multiprocessing

def process_row(args):
    row, nconf, energy, rms, seed, descr_num = args
    desc_mapping = DescMapping()
    molecule = Molecule(row['smiles'], row['mol_id'], row['activity'])
    molecule.gen_confs(nconf=nconf, energy=energy, rms=rms, seed=seed)
    desc_result = desc_mapping.calc_desc_mol(mol=molecule.mol, descr_num=descr_num)
    molecule.desc_result = desc_result
    return molecule, desc_result, desc_mapping

def map_desc(args):
    molecule, desc_mapping = args
    desc_result = molecule.desc_result
    new_desc_result = dict()
    for conf_id, descs in desc_result.items():
        new_desc_result[conf_id] = desc_mapping.load_conf_desc(descs)
    molecule.desc_result = new_desc_result
    return molecule

def mol_to_desc(smiles_data_path, save_path, nconf=5, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=10):
    smiles_data = pd.read_csv(smiles_data_path, names=['smiles', 'mol_id', 'activity'])
    desc_mapping = DescMapping()
    molecules = []

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    with multiprocessing.Pool(ncpu) as pool:
        args = [(row, nconf, energy, rms, seed, descr_num) for _, row in smiles_data.iterrows()]
        results = pool.map(process_row, args)
        molecules,desc_results,desc_mappings = zip(*results)

    for dm in desc_mappings:
        with lock:
            desc_mapping.merge(dm)


    with lock:

        desc_mapping.remove_desc()

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule, desc_mapping) for molecule in molecules]
        molecules = pool.map(map_desc, args)
        for molecule in molecules:
            molecule.load_conf_desc()
    
    desc_mapping.desc_mapping.to_csv(os.path.join(save_path, 'desc_mapping.csv'))

    with Chem.SDWriter(os.path.join(save_path, 'result.sdf')) as w:
        for m in molecules:
            w.write(m.mol)

    return desc_mapping, molecules

class Molecule:
    def __init__(self,smiles_str=None,mol_id=None,activity=None,mol=None):
        self.desc_result = dict()
        if mol is not None:
            self.mol = PropertyMol(mol)
            self.smiles_str = Chem.MolToSmiles(mol)
            self.mol_id = mol.GetProp("_Name")
            self.activity = mol.GetProp("Activity")
        else:
            self.smiles_str = smiles_str
            self.mol_id = mol_id
            self.activity = activity
            self.mol = PropertyMol(Chem.MolFromSmiles(smiles_str))
            self.mol.SetProp("_Name", str(mol_id))
            self.mol.SetProp("Activity", str(activity))
    def gen_confs(self,nconf=5, energy=100, rms=0.5, seed=42):
        self.mol = gen_confs_mol(mol=self.mol,nconf=nconf, energy=energy, rms=rms, seed=seed)
    def load_conf_desc(self):
        for conf in self.mol.GetConformers():
            conf.SetProp("Descriptors_index", json.dumps(self.desc_result[conf.GetId()]))
    pass


if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import rdchem

# 创建一个Mol对象
    mol = Chem.MolFromSmiles('CCO')

# 设置一个属性
    mol.SetProp("MyProperty", "MyValue")

# 将Mol对象序列化为一个pickle字符串
    mol_pickle = rdchem.Mol.ToBinary(mol)

# 将pickle字符串转换回Mol对象
    mol = rdchem.Mol(mol_pickle)

# 获取属性
    print(mol.GetProp("MyProperty"))  # 输出: MyValue