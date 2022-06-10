# -*- coding: utf-8 -*-
'''
@File    :   data_test.py
@Time    :   2022/04/20 10:29:37
@Author  :   Chen Huizhong
@Contact :   3190102098@zju.edu.cn
@Desc    :   None
'''

# Here put the import lib
from data_train_LSTM import model, device
from data_cache import test_X, test_Y
import matplotlib.pyplot as plt
import torch.nn as nn

model.eval()
test_Y_pred = model(test_X.to(device))

MSE = nn.MSELoss()(test_Y.to(device).float(), test_Y_pred.to(device).float())
print(f"测试集内的 MSE = {MSE}")

if __name__=='__main__':
    plt.figure(num='testing',figsize=(16,9))
    plt.plot(test_Y.detach().to('cpu').numpy(), label='observation')
    plt.plot(test_Y_pred.detach().to('cpu').numpy(), label='prediction')
    plt.title('Model testing on testing set', fontsize=20)
    plt.xlabel('data number', fontsize=20)
    plt.ylabel('significant wave height', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('testing.png')
    plt.show()

# 结果具有滞后性