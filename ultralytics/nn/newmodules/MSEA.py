import torch
from torch import nn

from ultralytics.nn.modules import C3

__all__ = ["MSEA", "C2f_MSEA", "C3k2_MSEA"]


class MSEA(nn.Module):  # MSEA多尺度特征增强聚合模块
    def __init__(self, dim, ratio=16):  # 初始化MSEA模块，dim是输入通道数，ratio用于通道注意力的压缩比
        super().__init__()
        self.dim_1 = dim  # 设置输入的通道数dim_1
        # 多尺度卷积分支
        self.branch_conv0 = nn.Conv2d(
            in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1
        )  # 1x1卷积，不改变通道数
        self.branch_conv1 = nn.Conv2d(
            in_channels=4 * self.dim_1 // 8,
            out_channels=4 * self.dim_1 // 8,
            stride=1,
            kernel_size=7,
            groups=4 * self.dim_1 // 8,
            dilation=3,
            padding=(1 + 6 * 3) // 2,
        )  # 7x7卷积，扩张率3，适合提取较大尺度的特征
        self.branch_conv2 = nn.Conv2d(
            in_channels=3 * self.dim_1 // 8,
            out_channels=3 * self.dim_1 // 8,
            stride=1,
            kernel_size=5,
            groups=3 * self.dim_1 // 8,
            dilation=2,
            padding=(1 + 4 * 2) // 2,
        )  # 5x5卷积，扩张率2，适合提取中等尺度特征
        self.branch_conv3 = nn.Conv2d(
            in_channels=self.dim_1 // 8,
            out_channels=self.dim_1 // 8,
            stride=1,
            kernel_size=3,
            groups=self.dim_1 // 8,
            dilation=1,
            padding=(1 + 2 * 1) // 2,
        )  # 3x3卷积，扩张率1，适合提取细节级别的特征
        self.P_conv = nn.Conv2d(
            in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1
        )  # 1x1卷积，用于最终输出的特征整合
        # 深度卷积分支
        self.avg_pool_1 = nn.AvgPool2d(7, 1, 3)  # 7x7平均池化，步长为1，padding为3，用于降维和捕捉全局信息

        self.conv1X1 = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_1, 1),  # 1x1卷积，保持通道数不变
            nn.BatchNorm2d(self.dim_1),  # 批归一化
            nn.ReLU(inplace=True),
        )  # ReLU激活函数

        self.depth_convs = nn.Sequential(
            nn.Conv2d(
                self.dim_1, self.dim_1, kernel_size=(1, 7), padding=(0, 7 // 2), groups=self.dim_1
            ),  # 1x7深度卷积
            nn.Conv2d(
                self.dim_1, self.dim_1, kernel_size=(7, 1), padding=(7 // 2, 0), groups=self.dim_1
            ),  # 7x1深度卷积
            nn.Conv2d(
                self.dim_1, self.dim_1, kernel_size=(1, 11), padding=(0, 11 // 2), groups=self.dim_1
            ),  # 1x11深度卷积
            nn.Conv2d(
                self.dim_1, self.dim_1, kernel_size=(11, 1), padding=(11 // 2, 0), groups=self.dim_1
            ),  # 11x1深度卷积
        )
        # 计算通道注意力
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将每个通道的特征图压缩为一个值
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化，用于获得通道的最大响应值

        self.fc1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)  # 1x1卷积，用于通道注意力的压缩
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)  # 1x1卷积，用于恢复通道注意力的大小
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成注意力权重

        self.act1 = nn.ReLU(inplace=True)  # ReLU激活函数
        self.act2 = nn.Sigmoid()  # Sigmoid激活函数，用于通道注意力的生成

    def forward(self, x):
        shortcut, x1, x2 = x, x, x  # 保留原始输入用于残差连接，复制x到x1和x2进行处理

        x1 = self.branch_conv0(x1)  # 对x1应用1x1卷积
        x1_1, x1_2, x1_3 = torch.split(
            x1, [4 * self.dim_1 // 8, 3 * self.dim_1 // 8, self.dim_1 // 8], dim=1
        )  # 分割x1为三个部分

        x1_1 = self.branch_conv1(x1_1)  # 对x1_1应用7x7卷积
        x1_2 = self.branch_conv2(x1_2)  # 对x1_2应用5x5卷积
        x1_3 = self.branch_conv3(x1_3)  # 对x1_3应用3x3卷积

        x1_out = self.act1(
            self.P_conv(torch.cat([x1_1, x1_2, x1_3], dim=1))
        )  # 将三个特征拼接并应用1x1卷积，最后通过ReLU激活

        x2_out = self.act2(
            self.conv1X1(self.depth_convs(self.conv1X1(self.avg_pool_1(x2))))
        )  # 通过深度卷积提取x2的特征，并应用ReLU和Sigmoid激活

        out = self.P_conv(x1_out * x2_out) + shortcut  # 将x1_out和x2_out进行逐元素乘法融合，并加上残差连接

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool_2(out))))  # 对out应用全局平均池化，然后通过通道注意力计算
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(out))))  # 对out应用全局最大池化，然后通过通道注意力计算

        ca_out = avg_out + max_out  # 将平均池化和最大池化的结果相加，得到最终的通道注意力
        output = out * self.sigmoid(ca_out)  # 将输出与通道注意力相乘，调整特征的重要性

        return output  # 返回最终的输出


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


class C2f_MSEA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MSEA(self.c) for _ in range(n))

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
        self.m = nn.Sequential(*(MSEA(c_) for _ in range(n)))


class C3k2_MSEA(C2f_MSEA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else MSEA(self.c) for _ in range(n))


# 输入 B C H W，输出 B C H W
if __name__ == "__main__":
    # 实例化模型对象
    model = MSEA(dim=32)  # 设置输入通道数为32
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)  # 随机生成一个输入张量，形状为(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)  # 将输入传入模型进行前向传播
    print("input_size:", input.size())  # 打印输入张量的大小
    print("output_size:", output.size())  # 打印输出张量的大小
