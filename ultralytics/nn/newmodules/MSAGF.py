import torch
from einops import rearrange
from torch import nn

__all__ = ["MSAGF"]


# 定义函数，将四维张量重排为三维
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")  # 将输入的四维张量重排为三维，合并高度和宽度维度


# 定义函数，将三维张量重排为四维
def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)  # 将三维张量恢复为四维，并重排维度


# PixelAttention类，用于计算像素级别的注意力
class PixelAttention(nn.Module):
    def __init__(self, dim):  # 初始化PixelAttention模块，dim为输入特征的通道数
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数，用于生成注意力图

    # 前向传播函数
    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W  # 在通道维度插入一个维度
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W  # 同样在pattn1插入一个维度
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W  # 沿着通道维度将x和pattn1拼接
        x2 = rearrange(x2, "b c t h w -> b (c t) h w")  # 重排张量维度为 [b, (c * t), h, w]
        pattn2 = self.pa2(x2)  # 经过卷积层计算注意力图
        pattn2 = self.sigmoid(pattn2)  # 通过Sigmoid归一化注意力图
        return pattn2  # 返回计算得到的注意力图


# 定义QKV_block类，用于计算多种池化操作后的特征
class QKV_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # 定义2x2池化层
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # 定义3x3池化层
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)  # 定义5x5池化层
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)  # 定义6x6池化层

    # 前向传播函数
    def forward(self, x):
        b, c, _h, _w = x.size()  # 获取输入x的形状，b为batch_size，c为通道数，h和w为高度和宽度
        pool_1 = self.pool1(x).view(b, c, -1)  # 对输入x进行2x2池化，并将输出展平
        pool_2 = self.pool2(x).view(b, c, -1)  # 对输入x进行3x3池化，并将输出展平
        pool_3 = self.pool3(x).view(b, c, -1)  # 对输入x进行5x5池化，并将输出展平
        pool_4 = self.pool4(x).view(b, c, -1)  # 对输入x进行6x6池化，并将输出展平
        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # 将所有池化结果拼接
        out = pool_cat.permute(0, 2, 1)  # 转置张量的维度为 [B,C,L]
        return out  # 返回计算结果


# 定义通道洗牌函数，用于打乱通道顺序
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的形状
    channels_per_group = num_channels // groups  # 计算每组的通道数
    x = x.view(batchsize, groups, channels_per_group, height, width)  # 重新调整张量的形状
    x = torch.transpose(x, 1, 2).contiguous()  # 转置通道组
    x = x.view(batchsize, -1, height, width)  # 展平通道并恢复张量的形状
    return x  # 返回洗牌后的张量


# 定义GLFA类，用于全局与局部特征的聚合
class GLFA(nn.Module):
    def __init__(self, in_channels):  # 初始化GLFA模块，in_channels为输入通道数
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.conv_1 = nn.Sequential(  # 定义第一层卷积层
            nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(  # 第二个卷积层
            nn.Conv2d(in_channels, in_channels, padding=2, kernel_size=3, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(  # 第三个卷积层
            nn.Conv2d(in_channels, in_channels, padding=3, kernel_size=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(  # 第四个卷积层
            nn.Conv2d(in_channels, in_channels, padding=4, kernel_size=3, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(  # 定义1x1卷积用于特征融合,调整通道数
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0
        )  # 1x1卷积调整输出通道

        self.SP_Pool_v = QKV_block(self.in_channels)  # 定义QKV块，用于计算值（V）
        self.SP_Pool_k = QKV_block(self.in_channels)  # 定义QKV块，用于计算键（K）
        nn.init.constant_(self.W.weight, 0)  # 初始化W卷积层的权重为0
        nn.init.constant_(self.W.bias, 0)  # 初始化W卷积层的偏置为0

    # 前向传播函数
    def forward(self, x):
        # 捕捉局部特征
        c1 = self.conv_1(x)  # 经过第一个卷积层
        c2 = self.conv_2(x)  # 经过第二个卷积层
        c3 = self.conv_3(x)  # 经过第三个卷积层
        c4 = self.conv_4(x)  # 经过第四个卷积层
        cat = torch.cat([c1, c2, c3, c4], dim=1)  # 将四个特征拼接
        fuse_out = self.fuse(cat)  # 融合特征
        out = self.W(fuse_out)  # 最后的1x1卷积层输出
        return out  # 返回最终输出


# MSAGF模块的定义：结合GLFA（全局与局部特征融合）与PixelAttention（像素级注意力机制）
class MSAGF(nn.Module):
    def __init__(self, in_dim, out_dim):  # 初始化MSAGF模块，dim为输入特征的通道数
        super().__init__()
        self.GLFA = GLFA(in_dim)  # 初始化GLFA模块
        self.PixelAttention = PixelAttention(in_dim)  # 初始化PixelAttention模块
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=True)  # 1x1卷积层，用于调整输出通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    # 前向传播函数
    def forward(self, data):
        x, y = data[0], data[1]
        initial = x + y  # 将输入x与y相加，作为初始特征
        pattn1 = self.GLFA(initial)  # 通过GLFA模块提取全局与局部特征
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))  # 通过PixelAttention计算像素级注意力
        result = initial + pattn2 * x + (1 - pattn2) * y  # 根据注意力图对x和y进行加权融合
        result = self.conv(result)  # 通过1x1卷积调整通道数
        return result  # 返回融合后的结果


# 测试代码
if __name__ == "__main__":
    block = MSAGF(32, 32)  # 创建MSAGF模块实例，输入通道数in_dim=32，输出通道数out_dim=32
    input1 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input1，大小为(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input2，大小为(1, 32, 64, 64)
    output = block(input1, input2)  # 将input1和input2传入MSAGF模块进行前向传播
    print("input1_size:", input1.size())  # 打印input1的尺寸
    print("output_size:", output.size())  # 打印输出的尺寸
