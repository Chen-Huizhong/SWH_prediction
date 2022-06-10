# -*- coding: utf-8 -*-
'''
@File    :   data_cache.py
@Time    :   2022/04/18 22:12:58
@Author  :   Chen Huizhong
@Contact :   3190102098@zju.edu.cn
@Desc    :   
'''

# Here put the import lib
from data_read import data47, data46
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

"""
TODO:
单层LSTM的预测结果存在滞后性。
可能的原因：序列本身存在自相关性，用上一步的结果作为下一步的预测结果本身误差就足够小。
消除自相关性，以解决滞后性。
"""

# 修改数据集
def change_shape(data, time_step):
    '''
    将数据集变成我们需要的样子。
    输入：
        data: 数据集，大小为 记录条数*特征数量
        time_step: 时间步数，前tau步预测后1步
    返回：
        temp: 变形后的数据集，大小为 (记录条数-时间步数)*(时间步数*特征数量)
    '''
    T = len(data)
    num_features = data.shape[1]
    temp = torch.zeros((T-time_step, time_step*num_features))
    _ = 0
    for i in range(time_step):
        temp[:, _:_+num_features] = data[i:T-time_step+i, :]
        _ += num_features
    return temp

# 超参数
time_step = 24  # 时间步长
batch_size = 48  # minibatch大小
train_len = int(len(data47)*0.75)

# 数据集扩充
wvht = np.array(data47['WVHT'])
wvht = torch.from_numpy(wvht).unsqueeze(1).float()

# 训练集数据
train_data = wvht[:train_len, :]
train_X = change_shape(train_data, time_step).reshape(-1, time_step, 1)
train_Y = train_data[time_step:, 0].unsqueeze(1)

# 测试集数据
test_data = wvht[train_len:, :]
test_X = change_shape(test_data, time_step).reshape(-1, time_step, 1)
test_Y = test_data[time_step:, 0].unsqueeze(1)

print(f'训练集数据大小{len(train_data)}')
print(f'测试集数据大小{len(test_data)}')

# 构建dataloader
train_dataset = TensorDataset(train_X, train_Y)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

'''
TODO:
    significant wave height 1 hour ago
    wind direction 2 hours ago
    wind speed 2 hours ago
    wind direction 1 hours ago
    wind speed 1 hours ago
    作为输入特征
'''