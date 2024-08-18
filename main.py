import os
import argparse
import torch
import torch.nn as nn
from data.dataset import MolDataSet,MolData,MolSoapData,TestData,MolSoapFlatData
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
from model.utils import dataset_split,scale_data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
#python main.py --ncpu 10 --device cuda --nconf 2
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help='the number of training epoch')
parser.add_argument('--patience', type=int, default=30, help='the patience of earlystop')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')   
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--instance_dropout', type=float, default=0.25, help='instance dropout')
parser.add_argument('--data_path', type=str, default='train') 
parser.add_argument('--nconf', type=int, default=5, help='conformers to generate for each molecule')
parser.add_argument('--ncpu', type=int, default=60, help='how many cpu to use for data processing')
parser.add_argument('--device', default='0,1,2,3,4,5', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args() 

# 定义当前模型的训练环境
# 
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 
batch_size = opt.batch_size
lr = opt.lr
weight_decay = opt.weight_decay
instance_dropout = opt.instance_dropout
data_path = os.path.join(os.getcwd(),'data','datasets',f'{opt.data_path}.csv')
save_path = os.path.join(os.getcwd(),'data','descriptors',f'{opt.data_path}')
epochs = opt.epochs
patience = opt.patience
nconf = opt.nconf
ncpu = opt.ncpu

def main(data_path,save_path,epochs,batch_size,lr,weight_decay,instance_dropout,nconf,ncpu,device):
    # 设置初始学习率和最终学习率
    initial_lr = 1e-6
    final_lr = 1e-1

    # 创建 TensorBoard 的 SummaryWriter
    writer = SummaryWriter(log_dir='logs/lr_range_test')
    # 加载数据集
    generator = torch.Generator().manual_seed(6)
    molData = MolSoapData(data_path,save_path,nconf=nconf, energy=100, rms=0.5, seed=42, ncpu=ncpu)
    train_dataset,test_dataset,val_dataset = molData.preprocess()
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    batch_amount = len(train_dataloader)
    num_iterations = batch_amount * epochs  # 总的训练步骤
    # 创建学习率调度器
    lrs = torch.logspace(start=np.log10(initial_lr), end=np.log10(final_lr), steps=num_iterations)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=True)
    # 初始化模型
    model = BagAttentionNet(ndim=(train_dataset[0][0][0].shape[1],256,128,64),det_ndim=(64,64),instance_dropout=instance_dropout).to(device)
    #double
    model = model.double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=weight_decay)

    # 初始化用于保存loss的列表
    train_losses = []
    val_losses = []
    test_losses = []

    # 早停
    earlystopping = EarlyStopping(patience=patience,verbose=True)

    # 训练模型
    for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_loss = 0
        train_progress = tqdm(enumerate(train_dataloader), desc="Batches", position=0, leave=True)
        for i,((bags,mask),labels) in train_progress:

            # 设置当前学习率
            lr = lrs[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            bags = bags.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            weight,outputs = model(bags,mask)
            loss = criterion(outputs, labels)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和学习率到 TensorBoard
            writer.add_scalar('Loss', loss.item(), i+epoch*batch_amount)
            writer.add_scalar('Learning Rate', lr, i+epoch*batch_amount)

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
            logging.info("Early stopping",f'loss: {avg_val_loss}')
            break
    loss_data = pd.DataFrame({'train_loss':train_losses,'val_loss':val_losses,'test_loss':test_losses})
    loss_data.to_csv(os.path.join(save_path,'loss.csv'),header=False,index=False)
    
    model.eval()
    with torch.no_grad():
        def eval_model(dataloader, model, device, save_path, file_name):
            progress = tqdm(enumerate(dataloader), desc=file_name, position=0, leave=True)
            weights = []
            y_pred = []
            y_label = []
            for i, ((bags, mask), labels) in progress:
                bags = bags.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                weight, outputs = model(bags, mask)
                w = weight.view(weight.shape[0], weight.shape[-1]).cpu()
                w = [i[j.bool().flatten()].detach().numpy() for i, j in zip(w, mask)]
                weights.extend(w)
                y_pred.extend(outputs.cpu().detach().numpy())
                y_label.extend(labels.cpu().detach().numpy())
            weights = np.array(weights)
            weight_data = pd.DataFrame(weights)
            weight_data.to_csv(os.path.join(save_path, f'{file_name}_weight.csv'))
            y_pred = np.array(y_pred)
            y_label = np.array(y_label)
            np.savetxt(os.path.join(save_path, f'{file_name}_pred.csv'), np.column_stack((y_label,y_pred)), delimiter=',')
            logging.info(f'R2 score {file_name}:{r2_score(y_label, y_pred)}')

        # 使用新的函数来进行训练、测试和验证
        eval_model(train_dataloader, model, device, save_path, 'train')
        eval_model(test_dataloader, model, device, save_path, 'test')
        eval_model(val_dataloader, model, device, save_path, 'val')
    pass
import matplotlib.pyplot as plt
import seaborn as sns
#绘制学习曲线
#平滑曲线
def smooth_curve(points,factor=0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+point*(1-factor))
        else:
            smooth_points.append(point)
    return smooth_points

def lr_curve():
    loss_result = np.loadtxt(open(os.path.join(save_path,'loss.csv'), 'rb'), delimiter=",")
    loss_result = loss_result.reshape(loss_result.shape[1],loss_result.shape[0])
    train_loss = smooth_curve(loss_result[0])
    test_loss = smooth_curve(loss_result[1])
    val_loss = smooth_curve(loss_result[2])


    plt.plot(range(10, len(train_loss)), train_loss[10:], label='train_loss', linestyle='--')  # 从第8个epoch绘图
    plt.plot(range(10, len(test_loss)), test_loss[10:], label='test_loss', linestyle='-')
    plt.plot(range(10, len(val_loss)), val_loss[10:], label='val_loss', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss_MAE')
    plt.grid()
    plt.legend()
    plt.savefig('Loss.png',dpi=1000)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')]  # 添加这一行
    )
    logging.info('------------start------------')
    main(data_path,save_path,epochs,batch_size,lr,weight_decay,instance_dropout,nconf,ncpu,device)
    lr_curve()
    pass