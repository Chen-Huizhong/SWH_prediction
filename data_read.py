# -*- coding: utf-8 -*-
'''
@File    :   data_readv2.py
@Time    :   2022/04/29 09:54:54
@Author  :   Chen Huizhong
@Contact :   3190102098@zju.edu.cn
@Desc    :   [Data resource] https://www.ndbc.noaa.gov/

数据描述：
    WDIR: Wind direction [col = 5] 
    WSPD: Wind speed
    GST : Peak 5 or 8 second gust speed
    WVHT: Significant wave hight [col = 8] 
          [missing_value = 99.00]
    DPD : Dominant wave period
          [missing_value = 99.00]
    APD : Average wave period
          [missing_value = 99.00]
    MWD : The direction from which the waves at the dominant period (DPD) are coming
          [missing_value = 999]
    PRES: Sea level pressure
   *ATMP: Air temperature (Celsius)
   *WTMP: Sea surface temperature 
   *DEWP: Dewpoint temperature taken at the same height as the air temperature measurement.
   *VIS : Station visibility
   *TIDE: The water level in feet above or below Mean Lower Low Water (MLLW).
'''

# Here put the import lib
from numpy import NaN
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def read_data(filepath, brokenLine=None):
    """
    从文件中读取数据。
    输入: 文件所在路径
    输出: DataFrame对象
    """
    # 数据读取
    data = pd.read_csv(filepath, sep='\s+', skiprows=[1], header=0)
    data.rename(columns={'#YY':'Date'}, inplace=True)

    # 日期格式转换
    data['Date'] = data['Date'].astype(str) + '/' + data['MM'].astype(str) + '/' +data['DD'].astype(str) \
        + ' ' + data['hh'].apply(lambda _: str(_).rjust(2, '0'))  
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d %H')  
    if brokenLine is None:
        # 删除无用数据
        data.drop(data[data['mm']!=40].index, axis=0, inplace=True)  
    else:
        # 删除无用数据
        data.drop(data[(data.index>=brokenLine) & (data['mm']!=40)].index, axis=0, inplace=True)
    # 删除无用列
    data.drop(['MM', 'DD', 'hh', 'mm', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE'], axis=1, inplace=True)  
    # 重新设置索引
    data = data.set_index('Date')

    return data

def interp_data(data):
    """
    对数据进行插值
    输入: 需要进行插值的数据
    输出: 插值后的数据
    """
    # 将缺失值设置为NaN
    data.replace([99., 999.], NaN, inplace=True)
    
    # 对数据进行时间补全并进行插值
    idx = pd.date_range(data.index[0], data.index[-1], freq='1H')
    data = data.reindex(idx)
    data.interpolate('cubic', inplace=True)

    return data

# 读取数据并插值

# Station41047的数据
data_2018 = read_data('./data/41047/41047h2018.txt')
data_2019 = read_data('./data/41047/41047h2019.txt')
data_2018 = interp_data(data_2018)
data_2019 = interp_data(data_2019)
data47 = pd.concat([data_2018, data_2019])
print(f'41047 总数据集大小{len(data47)}')

# Station41046的数据
data_2018 = read_data('./data/41046/41046h2018.txt')
data_2019 = read_data('./data/41046/41046h2019.txt')
data_2018 = interp_data(data_2018)
data_2019 = interp_data(data_2019)
data46 = pd.concat([data_2018, data_2019])
print(f'41046 总数据集大小{len(data46)}')


if __name__=='__main__':
    # 数据可视化
    plt.figure(num='data set', figsize=(16,9))
    a = plt.gca()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(data46.index, data46['WVHT'], label='41046')
    plt.plot(data47.index, data47['WVHT'], label='41047')
    train_len = int(len(data47)*0.75)
    # plt.axvline(data47.index[train_len], color=(1,0,0), linestyle='--')
    
    plt.xlim((data46.index[0]-120*data46.index.freq, \
        data46.index[-1]+120*data46.index.freq))
    plt.ylim((0, 10))
    a.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.legend(fontsize=20)
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.ylabel('Significant Wave Height (m)', fontsize=20)
    plt.xlabel('date', fontsize=20)

    plt.savefig('swh.png')
    plt.show()