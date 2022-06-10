# -*- coding: utf-8 -*-
'''
@File    :   data_train.py
@Time    :   2022/04/19 09:35:30
@Author  :   Chen Huizhong
@Contact :   3190102098@zju.edu.cn
@Desc    :   LSTM neural network
'''

# TODO: 升级网络模型

# Here put the import lib
from data_cache import train_X, train_Y, data_loader
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# 获取训练中所需要的设备
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
train_X = train_X.to(device)
train_Y = train_Y.to(device)

# 自定义LSTM神经网络
class MyLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyLstm, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 别忘了改变h0和c0的设备，它们不是由模型生成的参数，所以需要自己设定
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


input_dim = 1  # 输入的特征只有一个
hidden_dim = 48  # 隐藏层维度
num_layers = 1  # 隐藏层的个数
output_dim = 1  # 输出特征数量
learning_rate = 0.001  # 学习率
epochs = 50  # 循环次数

model = MyLstm(input_dim=input_dim, 
               hidden_dim=hidden_dim, 
               num_layers=num_layers, 
               output_dim=output_dim).to(device=device)
criterion = nn.MSELoss()  #TODO: 更换评测函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  #TODO: 更换学习方法

hist = torch.zeros(epochs)
start_time = time.time()

for t in range(epochs):
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        
        y_train_pred = model(X).to(device)

        loss = criterion(y_train_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_train_pred = model(train_X).to(device)
    loss = criterion(y_train_pred, train_Y)
    print(f"Epoch {t}, MSE: {loss.item()}")
    hist[t] = loss.item()

# 训练时间
training_time = time.time() - start_time
print(f"training time: {training_time}")

#TODO: 保存训练结果

if __name__ == '__main__':
    # 训练结果可视化
    plt.figure(num='training MSE', figsize=(16,9))
    plt.plot(hist)
    plt.ylabel('Mean Square Error while training', fontsize=20)
    plt.xlabel('Epoches', fontsize=20)
    plt.savefig('MSE.png')
    plt.show()