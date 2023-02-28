# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 2:44
# @Author  : Yujin Wang


import numpy as np
import pandas as pd

# 将数据转换为 numpy 数组
pd_data = pd.read_csv('./data/c800w.csv').to_numpy().squeeze()

data = pd_data

# 滑动窗口大小为 101，步长为 3
window_size = 101
stride = 3

# 创建 train_data 和 labels
train_data = []
labels = []

# 处理数据成滑动窗口形式
for i in range(0, len(data) - window_size + 1, stride):
    train_data.append(data[i:i+window_size-1])
    labels.append(data[i+window_size-1])

# 将列表转换为 numpy 数组
train_data = np.array(train_data)
labels = np.array(labels)

print("train_data shape:", train_data.shape)  # (31, 100)
print("labels shape:", labels.shape)  # (31,)