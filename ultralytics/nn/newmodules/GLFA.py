import torch
from torch import nn

from ultralytics.nn.modules import C3

__all__ = ["GLFA", "C2f_GLFA", "C3k2_GLFA"]


class QKV_block(nn.Module):  # 定义一个QKV_block类，继承自nn.Module
    def __init__(self, in_channels):  # 初始化函数，接受输入通道数，没有用到
        super().__init__()
        # 定义四个不同大小的最大池化操作
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # 2x2池化，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # 3x3池化，步幅为3
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)  # 5x5池化，步幅为5
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)  # 6x6池化，步幅为6

    def forward(self, x):  # 前向传播函数，输入x是一个4D张量(batch_size, channels, height, width)
        b, c, _h, _w = x.size()  # 获取输入x的尺寸，b为批量大小，c为通道数，h和w分别为高度和宽度
        pool_1 = self.pool1(x).view(b, c, -1)  # 对输入x进行2x2池化，并展平成(b, c, -1)，即将空间维度展平
        pool_2 = self.pool2(x).view(b, c, -1)  # 对输入x进行3x3池化，并展平成(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)  # 对输入x进行5x5池化，并展平成(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)  # 对输入x进行6x6池化，并展平成(b, c, -1)
        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # 在通道维度上将四个池化结果拼接
        out = pool_cat.permute(0, 2, 1)  # 交换维度，使得输出形状为(b, h*w, c)，即通道维度和空间维度交换
        return out  # 返回输出


def channel_shuffle(x, groups):  # 定义通道混排函数，输入x是一个4D张量，groups是分组数
    batchsize, num_channels, height, width = x.data.size()  # 获取输入x的尺寸
    channels_per_group = num_channels // groups  # 计算每个组内的通道数
    x = x.view(batchsize, groups, channels_per_group, height, width)  # 将通道维度重塑为(groups, channels_per_group)
    x = torch.transpose(x, 1, 2).contiguous()  # 交换第1维和第2维，通道组与每组内的通道进行交换
    x = x.view(batchsize, -1, height, width)  # 将混排后的张量展平为(b, num_channels, height, width)
    return x  # 返回混排后的结果


class GLFA(nn.Module):  # 定义GLFA类，继承自nn.Module
    def __init__(self, in_channels):  # 初始化函数，接受输入通道数
        super().__init__()
        self.in_channels = in_channels  # 设置输入通道数
        self.out_channels = in_channels  # 设置输出通道数，通常在此情况下输入输出通道相同
        # 定义四个卷积层，分别用于捕捉不同尺度的局部特征
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=2, kernel_size=3, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=3, kernel_size=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=4, kernel_size=3, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SP_Pool_v = QKV_block(self.in_channels)  # 初始化QKV_block模块，用于生成值（V）
        self.SP_Pool_k = QKV_block(self.in_channels)  # 初始化QKV_block模块，用于生成键（K）
        nn.init.constant_(self.W.weight, 0)  # 初始化权重为零
        nn.init.constant_(self.W.bias, 0)  # 初始化偏置为零

    def forward(self, x):  # 前向传播函数，输入x是一个4D张量(batch_size, channels, height, width)
        # 捕捉局部特征
        c1 = self.conv_1(x)  # 使用conv_1捕捉局部特征
        c2 = self.conv_2(x)  # 使用conv_2捕捉局部特征
        c3 = self.conv_3(x)  # 使用conv_3捕捉局部特征
        c4 = self.conv_4(x)  # 使用conv_4捕捉局部特征
        cat = torch.cat([c1, c2, c3, c4], dim=1)  # 将局部特征拼接起来

        cat = channel_shuffle(cat, groups=4)  # 对拼接后的特征进行通道混排
        local_f = self.fuse(cat)  # 使用1x1卷积融合局部特征
        # 捕捉全局特征
        batch_size, _h, _w = x.size(0), x.size(2), x.size(3)  # 获取输入的尺寸
        query = x.view(batch_size, self.in_channels, -1)  # 将输入展平为查询特征
        query = query.permute(0, 2, 1)  # 调整查询的维度，使其适应矩阵乘法
        key = self.SP_Pool_k(x)  # 通过QKV_block生成键（K）特征
        key = key.permute(0, 2, 1)  # 调整键的维度顺序
        value = self.SP_Pool_v(x)  # 通过QKV_block生成值（V）特征
        sim_map = torch.matmul(query, key)  # 计算查询和键的点积，得到相似度矩阵
        sim_map = (self.in_channels**-0.5) * sim_map  # 缩放相似度矩阵，避免数值不稳定
        sim_map = sim_map.softmax(dim=-1)  # 对相似度矩阵进行softmax归一化，得到注意力权重
        global_f = torch.matmul(sim_map, value)  # 计算加权值（V），得到全局特征
        global_f = global_f.permute(0, 2, 1).contiguous()  # 调整全局特征维度
        global_f = global_f.view(batch_size, self.in_channels, *x.size()[2:])  # 恢复全局特征的空间维度
        global_f = self.W(global_f) + x  # 使用卷积W处理全局特征，并与输入x进行残差连接

        return global_f + local_f  # 返回全局特征与局部特征的融合结果


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class C2f_GLFA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GLFA(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLFA(c_) for _ in range(n)))


class C3k2_GLFA(C2f_GLFA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else GLFA(self.c) for _ in range(n))


if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1, 64, 32, 32)
    # 创建一个GLFA实例
    GLFA = GLFA(in_channels=64)
    # 将输入特征图传递给 HIFA模块
    output = GLFA(input)
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")
