import torch
from torch import nn

from ultralytics.nn.modules import C3

__all__ = ["CMEA", "C2f_CMEA", "C3k2_CMEA"]


# 通道注意力模块 (Channel Attention Block)
class CAB(nn.Module):
    def __init__(self, in_planes, ratio=16):  # 初始化 CAB 类，输入通道数和比例（默认为 16）
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化，输出大小为 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化，输出大小为 1x1

        # 定义通道注意力机制中的两层卷积
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一层卷积，减少通道数
        self.relu1 = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二层卷积，恢复通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于输出权重

    def forward(self, x):  # 定义前向传播方法
        # 分别使用平均池化和最大池化进行处理
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 使用平均池化，然后两层卷积
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 使用最大池化，然后两层卷积

        out = avg_out + max_out  # 将两者相加，融合通道信息
        return self.sigmoid(out)  # 对结果进行 Sigmoid 激活，生成通道注意力


# 空间注意力模块 (Spatial Attention Block)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):  # 初始化 SAB 类，默认为 7x7 卷积核
        super().__init__()
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )  # 卷积操作，输入通道为 2，输出通道为 1
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):  # 定义前向传播方法
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 沿着通道维度求平均值，保持维度
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度求最大值，保持维度

        x = torch.cat([avg_out, max_out], dim=1)  # 将平均池化和最大池化的结果拼接起来
        x = self.conv1(x)  # 通过卷积提取空间特征
        return self.sigmoid(x)  # 使用 Sigmoid 激活，生成空间注意力


# CMEA (卷积多尺度增强注意力模块)
class CMEA(nn.Module):
    def __init__(self, in_channels, ratio=16):  # 初始化 CMEA 类，输入通道数和比例（默认为 16）
        super().__init__()
        self.in_channels = in_channels  # 保存输入通道数
        # 定义深度可分离卷积（DWConv）模块，用于多尺度卷积
        self.dwconvs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            # 3x3 深度可分离卷积
            nn.BatchNorm2d(self.in_channels),  # 批量归一化
            nn.ReLU6(inplace=True),
        )  # ReLU6 激活函数

        # 定义 1x1 卷积，用于通道融合
        self.conv1X1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义不同尺度的深度可分离卷积（3x3, 5x5, 7x7, 9x9）
        self.dwconv3X3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True),
        )
        self.dwconv5X5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 5 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True),
        )
        self.dwconv7X7 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 7, 1, 7 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True),
        )
        self.dwconv9X9 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 9, 1, 9 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True),
        )

        # 定义计算通道注意力的部分
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)  # 第一层卷积，降低通道数
        self.relu1 = nn.ReLU()  # ReLU 激活
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)  # 第二层卷积，恢复通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活

        # 定义计算空间注意力的卷积部分
        self.conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)  # 7x7 卷积，输入 2 个通道，输出 1 个通道
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活

    def forward(self, x):  # 定义前向传播方法
        # 多尺度卷积
        x = self.dwconv3X3(x)  # 3x3 卷积
        x = x + self.dwconv5X5(x) + self.dwconv7X7(x) + self.dwconv9X9(x)  # 加上其他尺度的卷积输出
        x = self.conv1X1(x)  # 1x1 卷积，调整通道数

        # 计算通道注意力
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化后通过全连接层
        max_out1 = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化后通过全连接层
        out1 = avg_out1 + max_out1  # 两者相加，融合信息
        c_attention = self.sigmoid(out1)  # 使用 Sigmoid 激活函数，得到通道注意力

        # 计算空间注意力
        avg_out2 = torch.mean(x, dim=1, keepdim=True)  # 沿着通道维度求平均
        max_out2, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度求最大值
        out2 = torch.cat([avg_out2, max_out2], dim=1)  # 拼接平均池化和最大池化的结果
        out2 = self.conv1(out2)  # 通过卷积获取空间注意力
        s_attention = self.sigmoid(out2)  # 使用 Sigmoid 激活函数，得到空间注意力

        # 最终输出：通道注意力和空间注意力加权后的特征
        output = x * c_attention + x * s_attention  # 加权融合
        return output  # 返回输出


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


class Bottleneck_CMEA(nn.Module):
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
        self.Attention = CMEA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C2f_CMEA(nn.Module):
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
            Bottleneck_CMEA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
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
        self.m = nn.Sequential(*(Bottleneck_CMEA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_CMEA(C2f_CMEA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_CMEA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )


# # 输入 B C H W,  输出 B C H W
# if __name__ == '__main__':
#     model = CMEA(in_channels=32)  # 实例化 CMEA 模型，输入通道数为 32
#     input = torch.randn(1, 32, 64, 64)  # 生成随机输入张量，形状为 [1, 32, 64, 64]
#     output = model(input)  # 执行前向传播
#     print('input_size:', input.size())  # 打印输入张量的形状
#     print('output_size:', output.size())  # 打印输出张量的形状
