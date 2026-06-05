
CIFAR格式数据集说明
===================

数据集信息
---------
- 来源: YOLO格式目标检测数据集转换
- 总样本数: 16866
- 训练集样本: 12470 (5个批次)
- 测试集样本: 4396 (1个批次)
- 类别数: 5
- 图像尺寸: 32×32×3

文件结构
-------
F:\contrast-h\yolov5-master\datasets\C-M1107\data_C-M1107\cifar-10-batches-py/
├── batches.meta           # 元数据文件
├── data_batch_1          # 训练批次1
├── data_batch_2          # 训练批次2
├── data_batch_3          # 训练批次3
├── data_batch_4          # 训练批次4
├── data_batch_5          # 训练批次5
└── test_batch           # 测试批次

类别分布
-------
- car: 9246个样本 (54.8%)
- motor: 1826个样本 (10.8%)
- truck: 2058个样本 (12.2%)
- van: 1629个样本 (9.7%)
- person: 2107个样本 (12.5%)

数据格式
-------
每个批次文件是一个Python pickle格式的字典，包含:
- 'data': numpy uint8数组，形状为(n, 3072)
- 'labels': 整数标签列表
- 'filenames': 文件名列表
- 'batch_label': 批次描述

每个样本是32×32的RGB图像，展平为3072维向量:
前1024个值是红色通道，中间1024个是绿色通道，最后1024个是蓝色通道。

使用方法
-------
# Python示例
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载一个批次
batch = unpickle('data_batch_1')
data = batch[b'data']  # 注意: 实际键是字节字符串
labels = batch[b'labels']

# 重塑为32x32x3图像
image = data[0].reshape(3, 32, 32).transpose(1, 2, 0)
