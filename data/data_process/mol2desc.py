from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem.PropertyMol import PropertyMol
from .gen_conf import gen_confs_mol, serialize_conf, deserialize_mol
from .data_utils import appendDataLine, divide_list
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
    logging.info(f'mol_to_desc> map_desc: mol_id: {molecule.mol_id}')
    desc_result = molecule.desc_result
    new_desc_result = dict()
    for conf_id, descs in desc_result.items():
        new_desc_result[conf_id] = desc_mapping.load_conf_desc(descs)
    molecule.desc_result = new_desc_result
    return molecule

def merge_desc(args):
    desc_mapping_result, sub_desc_mappings = args
    logging.info(f'mol_to_desc> merge_desc: {sub_desc_mappings.__len__()}')
    for sub_desc_mapping in sub_desc_mappings:
        desc_mapping_result.merge(sub_desc_mapping)
    return desc_mapping_result

def mol_to_desc(smiles_data_path, save_path, nconf=2, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=10,new=False):
    molecules = []

    if (not new) & os.path.exists(os.path.join(save_path, 'desc_mapping.csv')) & os.path.exists(os.path.join(save_path, 'result.sdf')):
        desc_mapping = DescMapping(pd.read_csv(os.path.join(save_path,'desc_mapping.csv')))
        supplier = Chem.SDMolSupplier(os.path.join(save_path,'result.sdf'))
        for mol in supplier:
            molecule = Molecule(mol=mol)
            molecule.load_desc_result_to_prop()
            molecule.load_conf_desc()
            molecules.append(molecule)
        return desc_mapping, molecules
    
    smiles_data = pd.read_csv(smiles_data_path, names=['smiles', 'mol_id', 'activity'])
    desc_mapping = DescMapping()

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    with multiprocessing.Pool(ncpu) as pool:
        args = [(row, nconf, energy, rms, seed, descr_num) for _, row in smiles_data.iterrows()]
        results = pool.map(process_row, args)
        molecules,desc_results,desc_mappings = zip(*results)

    desc_mapping_results = [DescMapping() for _ in range(ncpu)]

    with multiprocessing.Pool(ncpu) as pool:
        args = [(desc_mapping_results[index], sub_desc_mappings) for index,sub_desc_mappings in enumerate(divide_list(desc_mappings,ncpu))]
        desc_mappings = pool.map(merge_desc, args)

    for i,dm in enumerate(desc_mappings):
        logging.info(f'mol_to_desc> desc_mappings[{i}].merge')
        desc_mapping.merge(dm)

    desc_mapping.remove_desc()

    with multiprocessing.Pool(ncpu) as pool:
        args = [(molecule, desc_mapping) for molecule in molecules]
        molecules = pool.map(map_desc, args)
        for molecule in molecules:
            molecule.save_desc_result_to_prop()
            molecule.load_conf_desc()
    
    desc_mapping.desc_mapping.to_csv(os.path.join(save_path, 'desc_mapping.csv'))

    with Chem.SDWriter(os.path.join(save_path, 'result.sdf')) as w:
        for m in molecules:
            print('molprops>',list(m.mol.GetPropNames(includePrivate=True)))
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
    def gen_confs(self,nconf=2, energy=100, rms=0.5, seed=42):
        self.mol = gen_confs_mol(mol=self.mol,nconf=nconf, energy=energy, rms=rms, seed=seed)
    def load_desc_result_to_prop(self):
        if self.mol.HasProp("Descriptors_result"):
            self.desc_result = json.loads(self.mol.GetProp("Descriptors_result"))
    def save_desc_result_to_prop(self):
        self.mol.SetProp("Descriptors_result", json.dumps(self.desc_result))
        #不知道为什么这些Prop没有在第一次设置时保存，重新设置了一次又保存成功了
        self.mol.SetProp("_Name", str(self.mol_id))
        self.mol.SetProp("Activity", str(self.activity))
    #构象的Prop完全无法保存，不得不每次重新读取
    def load_conf_desc(self):
        for conf in self.mol.GetConformers():
            #Descriptors_result会在存入文件后由<int,int>转为<str,str>，因此需要判断desc_result的key类型
            #有些分子可能没有构象
            if self.desc_result.keys().__len__() == 0:
                continue
            if isinstance(list(self.desc_result.keys())[0],str):
                conf.SetProp("Descriptors_index", json.dumps(self.desc_result[str(conf.GetId())]))
            else:
                conf.SetProp("Descriptors_index", json.dumps(self.desc_result[conf.GetId()]))
    pass


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