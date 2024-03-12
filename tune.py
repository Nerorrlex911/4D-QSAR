import os
import argparse
import torch
import torch.nn as nn
from data.dataset import MolDataSet
from torch.utils.data import DataLoader,random_split
import logging
import sys
from model.BagAttentionNet import BagAttentionNet
from model.earlystop import EarlyStopping
import pickle
import numpy as np  
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score
from model.utils import dataset_split
import time
#python main.py --ncpu 10 --device cuda --nconf 2
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='the number of training epoch')
parser.add_argument('--lr', type=float, default=0.005, help='start learning rate')   
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--instance_dropout', type=float, default=0.95, help='instance dropout')
parser.add_argument('--data_path', type=str, default='train') 
parser.add_argument('--nconf', type=int, default=100, help='conformers to generate for each molecule')
parser.add_argument('--ncpu', type=int, default=60, help='how many cpu to use for data processing')
parser.add_argument('--device', default='0,1,2,3,4,5', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args() 

# 定义当前模型的训练环境python main.py --ncpu 30 --device 0,1,2,3,4,5
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 
lr = opt.lr
weight_decay = opt.weight_decay
instance_dropout = opt.instance_dropout
data_path = os.path.join('data','datasets',f'{opt.data_path}.csv')
save_path = os.path.join('data','descriptors',f'{opt.data_path}')
epochs = opt.epochs
nconf = opt.nconf
ncpu = opt.ncpu


def batch_size_test(dataset,generator,batch_size: int = 8):
    train_dataloader,test_dataloader,val_dataloader = dataset_split(dataset=dataset,train=0.7,val=0.2,batch_size=batch_size,generator=generator)
    # 初始化模型
    model = BagAttentionNet(ndim=(dataset[0][0][0].shape[1],128,64,64),det_ndim=(64,64),instance_dropout=instance_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=weight_decay)

    for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_progress = tqdm(enumerate(train_dataloader), desc="Batches", position=0, leave=True)
        for i,((bags,mask),labels) in train_progress:
            bags = bags.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            weight,outputs = model(bags,mask)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pass

def main():
    # 加载数据集
    generator = torch.Generator().manual_seed(42)
    dataset = MolDataSet(data_path,save_path,nconf=nconf, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=ncpu)   
    batch_sizes = [8,16,32,64,96]
    times = []
    for i in batch_sizes:
        start = time.time()
        batch_size_test(dataset=dataset,generator=generator,batch_size=i)
        end = time.time()
        times.append(end-start)
    print('time cost:   '+str(times))
    pass

if __name__ == "__main__":
    main()