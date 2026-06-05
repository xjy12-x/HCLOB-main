
CIFAR格式数据集说明（含背景样本）
=================================

数据集信息
---------
- 来源: YOLO格式目标检测数据集转换
- 转换方式: 从每张图片提取目标样本+随机背景样本
- 总样本数: 36516
- 训练集样本: 25526 (5个批次)
- 测试集样本: 10990 (1个批次)
- 类别数: 6 (目标类: 5, 背景类: 1)
- 图像尺寸: 32×32×3

文件结构
-------
F:\contrast-h\yolov5-master\MoCo\data_w\cifar-10-batches-py/
├── batches.meta           # 元数据文件
├── data_batch_1          # 训练批次1
├── data_batch_2          # 训练批次2
├── data_batch_3          # 训练批次3
├── data_batch_4          # 训练批次4
├── data_batch_5          # 训练批次5
└── test_batch           # 测试批次

类别分布
-------
- car: 10610个样本 (29.1%)
- motor: 2061个样本 (5.6%)
- truck: 2214个样本 (6.1%)
- van: 1752个样本 (4.8%)
- person: 2425个样本 (6.6%)
- background: 17454个样本 (47.8%)

背景样本生成
----------
- 每张图片尝试生成背景样本数: 3
- 背景与目标最大IoU阈值: 0.1
- 背景生成成功率: 17454/17454 (100.0%)

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
