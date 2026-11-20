import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import C3

__all__ = ["MKDA", "C2f_MKDA", "C3k2_MKDA"]


class LayerNorm(nn.Module):  # 定义一个 LayerNorm 类，继承自 nn.Module
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化可学习的偏置参数
        self.eps = eps  # 设置一个非常小的常数用于数值稳定性
        self.data_format = data_format  # 数据格式，支持 "channels_last" 或 "channels_first"
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果数据格式不正确，抛出异常
        self.normalized_shape = (normalized_shape,)  # 将 normalized_shape 转化为元组

    def forward(self, x):  # 定义前向传播函数
        if self.data_format == "channels_last":  # 如果数据格式为 "channels_last"
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )  # 使用 PyTorch 内置的 LayerNorm
        elif self.data_format == "channels_first":  # 如果数据格式为 "channels_first"
            u = x.mean(1, keepdim=True)  # 计算输入张量沿着通道维度的均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算输入张量的方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 标准化输入
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
            return x  # 返回标准化后的结果


class InceptionDWConv2d(nn.Module):  # 定义一个 Inception 风格的大核分解深度可分离卷积模块
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)  # 计算每个卷积分支的通道数
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )  # 定义深度可分离卷积，处理空间维度
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )  # 深度可分离卷积，处理宽度维度
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )  # 深度可分离卷积，处理高度维度
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)  # 计算拆分输入特征图的索引，分成 4 个部分

    def forward(self, x):  # 定义前向传播函数
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)  # 将输入张量拆分成 4 个部分
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )  # 将各个分支的卷积结果合并返回


class MKDA(nn.Module):  # 定义多尺度大核分解注意力（MKDA）模块
    def __init__(self, in_c):
        super().__init__()
        self.in_2c = 2 * in_c  # 定义输入通道数的 2 倍，用于通道扩展
        self.split_c = in_c // 4  # 将输入通道数分为 4 份
        self.i_c = self.split_c  # 设置每个分支的输入通道数
        self.norm = LayerNorm(in_c, data_format="channels_first")  # 实例化一个 LayerNorm 层
        self.scale = nn.Parameter(torch.zeros((1, in_c, 1, 1)), requires_grad=True)  # 定义一个可学习的缩放参数

        # 定义多个不同尺度的大核卷积
        self.LKA9 = nn.Sequential(  # 9x9 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=9),  # 使用 9x9 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0),
        )  # 使用 1x1 卷积调整输出通道
        self.LKA7 = nn.Sequential(  # 7x7 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=7),  # 使用 7x7 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0),
        )  # 使用 1x1 卷积调整输出通道
        self.LKA5 = nn.Sequential(  # 5x5 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=5),  # 使用 5x5 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0),
        )  # 使用 1x1 卷积调整输出通道
        self.LKA3 = nn.Sequential(  # 3x3 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=3),  # 使用 3x3 核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0),
        )  # 使用 1x1 卷积调整输出通道

        # 定义用于通道间的卷积操作，分别使用不同大小的卷积核
        self.k3 = nn.Conv2d(self.split_c, self.split_c, 3, 1, 1, groups=self.split_c)  # 3x3 卷积
        self.k5 = nn.Conv2d(self.split_c, self.split_c, 5, 1, 5 // 2, groups=self.split_c)  # 5x5 卷积
        self.k7 = nn.Conv2d(self.split_c, self.split_c, 7, 1, 7 // 2, groups=self.split_c)  # 7x7 卷积
        self.k9 = nn.Conv2d(self.split_c, self.split_c, 9, 1, 9 // 2, groups=self.split_c)  # 9x9 卷积

        # 用于调整输入通道数的 1x1 卷积操作
        self.proj_first = nn.Sequential(nn.Conv2d(in_c, self.in_2c, 1, 1, 0))  # 1x1 卷积，用于扩大通道数

        self.proj_last = nn.Sequential(  # 用于恢复原始通道数的 1x1 卷积
            nn.Conv2d(in_c, in_c, 1, 1, 0)
        )  # 1x1 卷积，用于还原输出通道数

    def forward(self, x):  # 定义前向传播函数
        shortcut = x  # 保存输入特征图，用于跳跃连接
        x = self.norm(x)  # 对输入进行层归一化
        x = self.proj_first(x)  # 通过 1x1 卷积扩展通道数

        a, x = torch.chunk(x, 2, dim=1)  # 将通道维度分成两部分，一部分用于注意力计算，另一部分继续处理
        a_1, a_2, a_3, a_4 = torch.chunk(a, 4, dim=1)  # 将注意力部分进一步拆分为 4 个部分

        # 计算各个卷积分支的特征，并加权融合
        a = torch.cat(
            [
                self.LKA3(a_1) * self.k3(a_1),
                self.LKA5(a_2) * self.k5(a_2),
                self.LKA7(a_3) * self.k7(a_3),
                self.LKA9(a_4) * self.k9(a_4),
            ],
            dim=1,
        )  # 各个卷积分支结果拼接

        # 通过注意力加权后的特征对 x 进行变换，并加上跳跃连接
        x = self.proj_last(x * a) * self.scale + shortcut  # 应用 1x1 卷积，结合缩放参数并加上原输入
        return x  # 返回最终输出


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


class Bottleneck_MKDA(nn.Module):
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
        self.Attention = MKDA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C2f_MKDA(nn.Module):
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
            Bottleneck_MKDA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
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
        self.m = nn.Sequential(*(Bottleneck_MKDA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_MKDA(C2f_MKDA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_MKDA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )


# 输入的形状为 B C H W，输出形状也是 B C H W
if __name__ == "__main__":
    model = MKDA(64).cuda()  # 实例化 MKDA 模型，并移动到 GPU
    input = torch.randn(1, 64, 32, 32).cuda()  # 生成随机输入，形状为 (1, 64, 32, 32)
    output = model(input)  # 执行前向传播
    print("input_size:", input.size())  # 打印输入的形状
    print("output_size:", output.size())  # 打印输出的形状
