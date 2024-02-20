import os
import argparse
import torch
import torch.nn as nn
from data.dataset import MolDataSet
from torch.utils.data import DataLoader,random_split
import logging
import sys
from model.BagAttentionNet import BagAttentionNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.005, help='start learning rate')   
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--instance_dropout', type=float, default=0.95, help='instance dropout')
parser.add_argument('--data_path', type=str, default='train') 
parser.add_argument('--nconf', type=int, default=5, help='conformers to generate for each molecule')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args() 

# 定义当前模型的训练环境
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 
batch_size = opt.batch_size
lr = opt.lr
weight_decay = opt.weight_decay
instance_dropout = opt.instance_dropout
data_path = os.path.join('data','datasets',f'{opt.data_path}.csv')
save_path = os.path.join('data','descriptors',f'{opt.data_path}')
epochs = opt.epochs
nconf = opt.nconf

def main(epochs,batch_size,lr,weight_decay,instance_dropout,nconf,device):
    # 加载数据集
    generator = torch.Generator().manual_seed(42)
    dataset = MolDataSet(data_path,save_path,nconf=nconf, energy=100, rms=0.5, seed=42, descr_num=[4])
    train_dataset,test_dataset,val_dataset = random_split(dataset=dataset,lengths=[0.7,0.2,0.1],generator=generator)
    logging.info(f'train_dataset:{len(train_dataset)} test_dataset:{len(test_dataset)} val_dataset:{len(val_dataset)}')
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False)
    # 初始化模型
    model = BagAttentionNet(ndim=dataset[0][0][0].shape[1],instance_dropout=instance_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=weight_decay)
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for i,(bags,labels) in enumerate(train_dataloader):
            bags = bags.to(device)
            labels = labels.to(device)
            weight,outputs = model(bags)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
        # 验证模型
        model.eval()
        with torch.no_grad():
            for bags,labels in val_dataloader:
                bags = bags.to(device)
                labels = labels.to(device)
                weight,outputs = model(bags)
                loss = criterion(outputs, labels)
                if i % 10 == 0:
                    logging.info(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {loss.item():.4f}')
    pass

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')]  # 添加这一行
    )
    logging.info('------------start------------')
    main(epochs,batch_size,lr,weight_decay,instance_dropout,nconf,device)
    pass