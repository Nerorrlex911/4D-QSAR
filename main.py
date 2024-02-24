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

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.005, help='start learning rate')   
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--instance_dropout', type=float, default=0.95, help='instance dropout')
parser.add_argument('--data_path', type=str, default='train') 
parser.add_argument('--nconf', type=int, default=5, help='conformers to generate for each molecule')
parser.add_argument('--ncpu', type=int, default=60, help='how many cpu to use for data processing')
parser.add_argument('--device', default='0,1,2,3,4,5', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args() 

# 定义当前模型的训练环境python main.py --ncpu 30 --device 0,1,2,3,4,5
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 
batch_size = opt.batch_size
lr = opt.lr
weight_decay = opt.weight_decay
instance_dropout = opt.instance_dropout
data_path = os.path.join('data','datasets',f'{opt.data_path}.csv')
save_path = os.path.join('data','descriptors',f'{opt.data_path}')
epochs = opt.epochs
nconf = opt.nconf
ncpu = opt.ncpu

def main(data_path,save_path,epochs,batch_size,lr,weight_decay,instance_dropout,nconf,ncpu,device):
    # 加载数据集
    generator = torch.Generator().manual_seed(42)
    dataset = MolDataSet(data_path,save_path,nconf=nconf, energy=100, rms=0.5, seed=42, descr_num=[4],ncpu=ncpu)
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    test_len = int(total_len * 0.2)
    val_len = total_len - train_len - test_len
    train_dataset,test_dataset,val_dataset = random_split(dataset=dataset,lengths=[train_len,test_len,val_len],generator=generator)
    logging.info(f'train_dataset:{len(train_dataset)} test_dataset:{len(test_dataset)} val_dataset:{len(val_dataset)}')
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False)
    # 初始化模型
    model = BagAttentionNet(ndim=(dataset[0][0][0].shape[1],128,64,64),det_ndim=(64,64),instance_dropout=instance_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=weight_decay)

    # 初始化用于保存loss的列表
    train_losses = []
    val_losses = []
    test_losses = []

    # 早停
    earlystopping = EarlyStopping(patience=30,verbose=True)

    # 训练模型
    for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_loss = 0
        train_progress = tqdm(enumerate(train_dataloader), desc="Batches", position=0, leave=True)
        for i,((bags,mask),labels) in train_progress:
            bags = bags.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            weight,outputs = model(bags,mask)
            loss = criterion(outputs, labels)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
        train_losses.append(train_loss/len(train_dataloader))
        # 验证模型
        model.eval()
        val_loss = 0
        test_loss = 0
        with torch.no_grad():
            for (bags,mask),labels in val_dataloader:
                bags = bags.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                weight,outputs = model(bags,mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                if i % 10 == 0:
                    logging.info(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {loss.item():.4f}')
            for (bags,mask),labels in test_dataloader:
                bags = bags.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                weight,outputs = model(bags,mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                if i % 10 == 0:
                    logging.info(f'Epoch [{epoch + 1}/{epochs}], Test Loss: {loss.item():.4f}')
        avg_val_loss = val_loss/len(val_dataloader)
        val_losses.append(avg_val_loss)
        test_losses.append(test_loss/len(test_dataloader))

        earlystopping(avg_val_loss,model)
        if earlystopping.early_stop:
            logging.info("Early stopping")
            #TODO:保存模型为pickle文件
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            break
    loss_data = pd.DataFrame({'train_loss':train_losses,'val_loss':val_losses,'test_loss':test_losses})
    loss_data.to_csv(os.path.join(save_path,'loss.csv'))
    weight,outputs = model(dataset.bags,dataset.mask)
    np.savetxt(os.path.join(save_path,'weights.csv'),weight.cpu().detach().numpy(),delimiter=',')
    logging.info(f'R2 score:{r2_score(dataset.labels.cpu().detach().numpy(),outputs.cpu().detach().numpy())}')
    pass

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')]  # 添加这一行
    )
    logging.info('------------start------------')
    main(data_path,save_path,epochs,batch_size,lr,weight_decay,instance_dropout,nconf,ncpu,device)
    pass