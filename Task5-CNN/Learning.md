## 模型训练
```python
# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
 
import time, datetime
import pdb, traceback
 
import cv2
# import imagehash
from PIL import Image
 
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
 
from efficientnet_pytorch import EfficientNet  
# model = EfficientNet.from_pretrained('efficientnet-b4')  #加载预训练EfficientNet
 
import torch
torch.manual_seed(0)  #随机初始化神经网络参数种子
torch.backends.cudnn.deterministic = False #优化运行效率
torch.backends.cudnn.benchmark = True  #优化运行效率
 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F   
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
 
 
    
class QRDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        print(img.shape)
        return img,torch.from_numpy(np.array(int('PNEUMONIA' in self.img_path[index])))  
    
    def __len__(self):
        return len(self.img_path)
    
 
 
#模型搭建 
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        
                
        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model
        
    def forward(self, img):   #模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
        out = self.resnet(img)
        return out
        
 
   
def train(train_iter, model, criterion, optimizer, scheduler,device):
    # 保存训练一个epoch的Loss值与准确率
    train_loss = 0.0
    train_acc = 0.0
    # 指定模型训练
    model.train()
    for i, (inputs, targets) in enumerate(train_iter):
        inputs = inputs.to(device)
        targets =targets.to(device)
        # 模型前向运行
        outputs = model(inputs)
        # 计算Loss值
        loss = criterion(outputs, targets)
        # 计算预测结果
        pred = outputs.argmax(dim=1)
        # 清除梯度
        optimizer.zero_grad()
        # 反传loss
        loss.backward()
        # 更新模型权重
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 统计Loss,Acc值
        train_loss += loss.item() * inputs.size(0)
        train_acc += (preds == targets.data).float().sum().item()
    epoch_loss = train_loss / train_iter.size(0)
    epoch_acc = train_acc / train_iter.size(0)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
 
    return model
 
# 定义验证阶段
def val(valid_iter, model, criterion):
    # 模型验证
    model.eval()
    # 指定不保存梯度
    with torch.no_grad():
        # 统计Loss值与准确率
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, targets) in enumerate(valid_iter):
            inputs = inputs.to(device)
            targets =targets.to(device)
            # 模型前向运行
            outputs = model(inputs)
            # 计算Loss值
            loss = criterion(outputs, targets)
            # 计算预测结果
            pred = outputs.argmax(dim=1)
            # 统计Loss,Acc值
            valid_loss += loss.item() * inputs.size(0)
            valid_acc += (preds == targets.data).float().sum().item()
        epoch_loss = valid_loss / dataset_sizes[phase]
        epoch_acc = float(running_corrects) / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('valid', epoch_loss, epoch_acc))
        return epoch_acc
 
def predict(test_loader, model):
    # switch to evaluate mode
    model.eval()     #设置模型在预测模式
    test_pred = []
    with torch.no_grad():        #with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
        end = time.time()
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)           #将tensor变成GPU上运行的tensor
            # compute output
            output = model(input)              
            output = output.data.to(device).numpy()
            test_pred.append(output)
        test_pred = np.vstack(test_pred)           #将list中的两个数组按纵向拼接成一个数组
                                                  #[[1,2,3],[3,4,5]]变成[1,2,3,3,4,5]
    return test_pred_tta
 
 
def train_model(train_jpg, model,criterion, optimizer, scheduler,K_flod=10,epochs=25,batch_size=256,model_save_path)
    kflod = KFold(n_splits=K_flod, random_state=233, shuffle=True)
    for flod_idx, (tr_idx, val_idx) in enumerate(kfold.split(train_jpg)):
        train_loader = torch.utils.data.DataLoader(QRDataset(train_jpg[tr_idx]
                       ,data_transforms['train'])
                       ,batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader   = torch.utils.data.DataLoader(QRDataset(train_jpg[val_idx]
                       ,data_transforms['valid']
                       ,batch_size=batch_size, shuffle=False, pin_memory=True)
        best_acc = 0.0
        for epoch in range(epochs):                          
            print('Epoch: ', epoch)
            model=train(train_loader, model, criterion, optimizer, scheduler,device)
            val_acc = val(val_loader, model, criterion,device)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path.format(flod_idx))
        print('Best val Acc: {:4f}'.format(best_acc))
 
data_transforms = {
    'train': transforms.Compose([
        # 随机在图像上裁剪出224*224大小的图像
        transforms.RandomResizedCrop(224),
        # 将图像随机翻转
        transforms.RandomHorizontalFlip(),
        # 将图像数据,转换为网络训练所需的tensor向量
        transforms.ToTensor(),
        # 图像归一化处理
        # 个人理解,前面是3个通道的均值,后面是3个通道的方差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
 
if __name__ ==  '__main__':
    # input dataset
    train_jpg = glob.glob('../input/xray_dataset/train/*/*')
    train_jpg = np.array(train_jpg)
    
    test_jpg= glob.glob("###################")
    test_jpg =np.array(test_jpg)
    
    #实例化模型
    model = VisitNet().to(device)
    #交叉熵能够表征真实样本标签和预测概率之间的差值
    criterion = nn.CrossEntropyLoss().to(device)        
    #CrossEntropyLoss() 会把target变成ont-hot形式
    optimizer = torch.optim.SGD(model.parameters(), 0.01)     
    #lr_scheduler调整策略：根据训练次数,optimizer 要更改学习率的优化器,step_size=4,每4轮更新一次学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85) #这里用的变学习率
    
    K_flod=10
    epochs=25
    batch_size=256
    
    model_save_path='./model_fold{0}.pt'
    
    train_model(train_jpg, model,criterion, optimizer, scheduler,K_flod,epochs,batch_size,model_save_path)
    
    
    model_list=glob.glob(model_save_path)
 
 
 
    for model_path in model_list:
        test_loader = torch.utils.data.DataLoader(QRDataset(test_jpg,data_transforms['test'])
            , batch_size=10, shuffle=False, num_workers=10, pin_memory=True)
        
        model = VisitNet().to(device)            #初始化模型
        model.load_state_dict(torch.load(model_path))  #将模型参数加载到模型中去
        # model = nn.DataParallel(model).cuda()
        test_pred = predict(test_loader, model)
    test_pred=test_pred/len(model_list)   
    
    test_csv = pd.DataFrame()
    test_csv[0] = list(range(1, len(test_jpg)))
    test_csv[1] = np.argmax(test_pred, 1)
    test_csv.to_csv('tmp.csv', index=None, header=None)
 
    
    
    
    
####################################    
#注           
#pytorch 1.10版本之前 应将scheduler.step()放在optimizer.step()之前
#pytorch 1.10版本之后 应将scheduler.step()放在optimizer.step()之后
                        
#torch.optim.lr_scheduler.LambdaLR()可自定义更新学习率的函数            
#scheduler= LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))        
```


## 经典模型搭建
```python

import time
import torch
from torch import nn, optim
import torchvision
import numpy as np
import sys
import os
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
class Reshape(nn.Module): #将图像大小重定型
    def __init__(self,shape_size=(-1,1,28,28)):
        super(Reshape,self).__init__()
        self.shape_size=shape_size
    def forward(self, x):
        return x.view(self.shape_size)      #(B x C x H x W)
        
class LeNet(nn.Module):
    def __init__(self,num_class=10):
        super(LeNet,self).__init__()
        self.num_class=num_class
        self.net = torch.nn.Sequential(     #Lelet                                                  
            Reshape(),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
            nn.Sigmoid(),                                                       
            nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
            Flatten(),                                                          #b*16*5*5   => b*400
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_class)
        )
    def forward(self,img):
        if ((torch.tensor(img.shape[2:])!=torch.tensor([28, 28]))[0].item()):
            print("请输入正确的图片尺寸：(m,*,28,28)")
        outpus=self.net(img)
        return out_puts
        
        
#AlexNet
# AlexNet
# 首次证明了学习到的特征可以超越⼿⼯设计的特征，从而⼀举打破计算机视觉研究的前状。
# 特征：
 
# 8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
# 将sigmoid激活函数改成了更加简单的ReLU激活函数。
# 用Dropout来控制全连接层的模型复杂度。
# 引入数据增强，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。
 
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            #FlattenLayer(),
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #由于使用CPU镜像，精简网络，若为GPU镜像可添加该层
            #nn.Linear(4096, 4096),
            #nn.ReLU(),
            #nn.Dropout(0.5),
 
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )
        # self.net.add_module('conv',self.conv)
        # self.net.add_module('fc',self.fc)
        
 
    def forward(self, img):
        if ((torch.tensor(img.shape[2:])!=torch.tensor([224, 224]))[0].item()):
            print("请输入正确的图片尺寸：(m,*,224,224)")
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        #output=self.net(img)
        return output
 
 
 
        
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
        
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
 
    
class VggNet (nn.Module):
    def __init__(self,conv_arch=((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)),
                                 fc_features=512 * 7 * 7,fc_hidden_units=4096):
        super(VggNet, self).__init__() 
        self.conv_arch = conv_arch
        # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
        self.fc_features = fc_features # c * w * h
        self.fc_hidden_units = fc_hidden_units# 任意
        
        self.net = nn.Sequential()
        # 卷积层部分
        for i, (num_convs, in_channels, out_channels) in enumerate(self.conv_arch):
            # 每经过一个vgg_block都会使宽高减半
            self.net.add_module("vgg_block_" + str(i+1), self.vgg_block(num_convs, in_channels, out_channels))
        # 全连接层部分
        self.net.add_module("fc", nn.Sequential(FlattenLayer(),
                             nn.Linear(self.fc_features, self.fc_hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(self.fc_hidden_units, self.fc_hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(self.fc_hidden_units, 10)
                            ))
        
    def vgg_block(self,num_convs, in_channels, out_channels): #卷积层个数，输入通道数，输出通道数
        blk = []
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
        return nn.Sequential(*blk)   
        
    def forward(self, img):
        if ((torch.tensor(img.shape[2:])!=torch.tensor([224, 224]))[0].item()):
            print("请输入正确的图片尺寸：(m,*,224,224)")
        output=self.net(img)
        return output
        
    
class NiNNet(nn.Module):
    def __init__(self,num_class=10):
        super(NiNNet, self).__init__()
        self.num_class=num_class
 
        self.net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Dropout(0.5),
            # 标签类别数是10
            self.nin_block(384, num_class, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d(), 
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            FlattenLayer()
            )
        
        
    def nin_block(self,in_channels, out_channels, kernel_size, stride, padding):
        blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=1),
                            nn.ReLU())
        return blk
    
 
    def forward(self,img):
        if ((torch.tensor(img.shape[2:])!=torch.tensor([224, 224]))[0].item()):
            print("请输入正确的图片尺寸：(m,*,224,224)")
        outputs=self.net(img)
        return outputs
 
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
 
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
        
class GoogLeNet(nn.Module):
 
    def __init__(self,num_class=10):
        super(GoogLeNet, self).__init__()
        
        self.num_class=num_class
        
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
 
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
 
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
 
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
 
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           GlobalAvgPool2d())
                           
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, FlattenLayer(), nn.Linear(1024, self.num_class))
        
    def forward(self,img):
        if ((torch.tensor(img.shape[2:])!=torch.tensor([96, 96]))[0].item()):
            print("请输入正确的图片尺寸：(m,*,96,96)")
        outputs=self.net(img)
        return outputs
 
class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    #可以设定输出通道数、是否使用额外的1x1卷积层来修改通道数以及卷积层的步幅。
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
 
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
    
class ResNet(nn.Module):
    def __init__(self,in_channels=64,num_class=10):
        super(ResNet,self).__init__()
        self.in_channels=in_channels
        self.num_class=num_class
        self.net=nn.Sequential()
        self.net.add_module("resnet_block1", self.resnet_block(self.in_channels, 64, 2, first_block=True))
        self.net.add_module("resnet_block2", self.resnet_block(64, 128, 2))
        self.net.add_module("resnet_block3", self.resnet_block(128, 256, 2))
        self.net.add_module("resnet_block4", self.resnet_block(256, 512, 2))
        self.net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
        self.net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, self.num_class))) 
 
    
    def resnet_block(self,in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)    

```
