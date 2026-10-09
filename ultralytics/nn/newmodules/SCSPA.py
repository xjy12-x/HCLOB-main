import torch
import torch.nn.functional
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import C3

__all__ = ["SCSPA", "C2f_SCSPA", "C3k2_SCSPA"]


class PixelAttention(nn.Module):  # 定义 PixelAttention 类，实现简单像素级别的注意力机制
    def __init__(self, channel):  # 初始化 PixelAttention 类，channel 表示输入的通道数
        super().__init__()
        self.pa = nn.Sequential(  # 定义一个由多个层组成的顺序模块
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),  # 1x1 卷积，通道数缩小为原来的 1/8
            nn.ReLU(inplace=True),  # ReLU 激活函数，激活每个元素
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),  # 1x1 卷积，将通道数压缩为 1
            nn.Sigmoid(),  # Sigmoid 激活函数，将输出限制在 [0, 1] 之间
        )

    def forward(self, x):  # 定义前向传播方法
        y = self.pa(x)  # 通过 Pixel Attention 模块计算像素级别的注意力图
        return x * y  # 将输入特征图 x 和注意力图 y 逐元素相乘，返回加权后的特征图


class ChannelAttention(nn.Module):  # 定义 ChannelAttention 类，实现通道级别的注意力机制
    def __init__(
        self, input_channels, internal_neurons
    ):  # 初始化 ChannelAttention 类，input_channels 为输入通道数，internal_neurons 为中间神经元数
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True
        )  # 1x1 卷积，降低通道数到 internal_neurons
        self.fc2 = nn.Conv2d(
            in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True
        )  # 1x1 卷积，恢复通道数到 input_channels
        self.input_channels = input_channels  # 保存输入的通道数，用于调整输出形状

    def forward(self, inputs):  # 定义前向传播方法
        x1 = F.adaptive_avg_pool2d(
            inputs, output_size=(1, 1)
        )  # 对输入进行自适应平均池化，输出为 (1, 1) 尺寸，压缩空间维度
        x1 = self.fc1(x1)  # 通过 fc1 卷积进行特征变换
        x1 = F.relu(x1, inplace=True)  # 使用 ReLU 激活函数
        x1 = self.fc2(x1)  # 通过 fc2 卷积恢复特征维度
        x1 = torch.sigmoid(x1)  # 使用 Sigmoid 激活函数生成通道注意力图
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))  # 对输入进行自适应最大池化，输出为 (1, 1)
        x2 = self.fc1(x2)  # 通过 fc1 卷积进行特征变换
        x2 = F.relu(x2, inplace=True)  # 使用 ReLU 激活函数
        x2 = self.fc2(x2)  # 通过 fc2 卷积恢复特征维度
        x2 = torch.sigmoid(x2)  # 使用 Sigmoid 激活函数生成通道注意力图
        x = x1 + x2  # 将两个通道注意力图相加，融合平均池化和最大池化的特征
        x = x.view(-1, self.input_channels, 1, 1)  # 调整输出形状为 (B, C, 1, 1)，以便与输入特征图逐元素相乘
        return x  # 返回通道注意力图


class SpatialAttention(nn.Module):  # 定义 SpatialAttention 类，实现空间级别的注意力机制
    def __init__(self, in_channels):  # 初始化 SpatialAttention 类，in_channels 为输入的通道数
        super().__init__()
        self.dconv5_5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels
        )  # 使用 5x5 深度可分离卷积
        self.dconv1_7 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels
        )  # 使用 1x7 深度可分离卷积
        self.dconv7_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels
        )  # 使用 7x1 深度可分离卷积
        self.dconv1_11 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels
        )  # 使用 1x11 深度可分离卷积
        self.dconv11_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels
        )  # 使用 11x1 深度可分离卷积
        self.dconv1_21 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels
        )  # 使用 1x21 深度可分离卷积
        self.dconv21_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels
        )  # 使用 21x1 深度可分离卷积
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)  # 使用 1x1 卷积融合多尺度特征

    def forward(self, inputs):  # 定义前向传播方法
        x_init = self.dconv5_5(inputs)  # 通过 5x5 深度可分离卷积提取空间特征
        x_1 = self.dconv1_7(x_init)  # 通过 1x7 深度可分离卷积提取空间特征
        x_1 = self.dconv7_1(x_1)  # 通过 7x1 深度可分离卷积提取空间特征
        x_2 = self.dconv1_11(x_init)  # 通过 1x11 深度可分离卷积提取空间特征
        x_2 = self.dconv11_1(x_2)  # 通过 11x1 深度可分离卷积提取空间特征
        x_3 = self.dconv1_21(x_init)  # 通过 1x21 深度可分离卷积提取空间特征
        x_3 = self.dconv21_1(x_3)  # 通过 21x1 深度可分离卷积提取空间特征
        x = x_1 + x_2 + x_3 + x_init  # 将所有卷积得到的特征加和，融合多尺度空间特征
        spatial_att = self.conv1x1(x)  # 使用 1x1 卷积融合所有空间特征，得到空间注意力图
        return spatial_att  # 返回空间注意力图


class SCSPA(nn.Module):  # SCSPA 协同通道空间像素注意力模块
    def __init__(
        self, in_channels, ratio=4
    ):  # 初始化 SCSPA 类，in_channels 为输入通道数，ratio 为通道注意力中间神经元的比例
        super().__init__()
        self.ca = ChannelAttention(
            input_channels=in_channels, internal_neurons=in_channels // ratio
        )  # 初始化通道注意力模块
        self.sa = SpatialAttention(in_channels)  # 初始化空间注意力模块
        self.pa = PixelAttention(in_channels)  # 初始化像素注意力模块

    def forward(self, inputs):  # 定义前向传播方法
        channel_att = self.ca(inputs)  # 计算通道注意力图
        out_ca = channel_att * inputs  # 将通道注意力图与输入特征图逐元素相乘，得到加权后的特征图

        Spatial_att = self.sa(out_ca)  # 计算空间注意力图

        Pixel_att = self.pa(out_ca)  # 计算像素注意力图

        out = Spatial_att * Pixel_att  # 将空间和像素注意力图逐元素相乘，得到最终加权特征图
        return out  # 返回加权后的特征图


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


class Bottleneck_SCSPA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = SCSPA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C2f_SCSPA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_SCSPA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

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
        self.m = nn.Sequential(*(Bottleneck_SCSPA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_SCSPA(C2f_SCSPA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_SCSPA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = SCSPA(in_channels=64, ratio=4)  # 创建 SCSPA 模块实例，输入通道数为 64，通道注意力比例为 4
    input_tensor = torch.randn(1, 64, 128, 128)  # 创建一个形状为 (1, 64, 128, 128) 的随机输入张量
    output_tensor = module(input_tensor)  # 通过 SCSPA 模块计算输出
    print("Input size:", input_tensor.size())  # 打印输入张量的形状
    print("Output size:", output_tensor.size())  # 打印输出张量的形状
