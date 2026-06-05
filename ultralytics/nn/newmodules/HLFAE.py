import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from ultralytics.nn.modules import C3
__all__=['HLFAE','C2f_HLFAE','C3k2_HLFAE']

# 特征提取模块
class feature_extraction(nn.Module):
    def __init__(self, in_channels):
        super(feature_extraction, self).__init__()
        # 1x1卷积，膨胀率为1
        self.dilate1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),  # 卷积层，膨胀率为1
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )
        # 1x1卷积，膨胀率为3
        self.dilate2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3),  # 卷积层，膨胀率为3
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )
        # 1x1卷积，膨胀率为5
        self.dilate3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5),  # 卷积层，膨胀率为5
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )
        # 1x1卷积，膨胀率为7
        self.dilate4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7),  # 卷积层，膨胀率为7
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )
        # 对四种不同膨胀卷积的输出进行通道拼接，进行通道压缩
        self.convmlp = nn.Sequential(
            nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=3, padding=1),  # 卷积层
            nn.BatchNorm2d(2 * in_channels),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),  # 卷积层，降维
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )

    def forward(self, x):
        # 计算四种不同膨胀卷积的输出
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(x)
        dilate3_out = self.dilate3(x)
        dilate4_out = self.dilate4(x)
        # 拼接四个不同膨胀卷积的输出
        cnn_out = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        # 使用1x1卷积进行通道压缩
        out = self.convmlp(cnn_out)
        return out

# 结合卷积、批归一化与ReLU激活的模块
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 是否使用ReLU激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),  # 卷积层
            nn.BatchNorm2d(out_c)  # 批归一化
        )
        self.relu = nn.ReLU(inplace=True)  # 激活函数

    def forward(self, x):
        x = self.conv(x)  # 卷积+批归一化
        if self.act == True:  # 如果设置了激活，则进行ReLU激活
            x = self.relu(x)
        return x

# Squeeze-and-Excitation (SE) 注意力机制模块
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层用于计算通道注意力
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一层FC
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二层FC
            nn.Sigmoid()  # Sigmoid激活，输出通道注意力权重
        )

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 初始化卷积层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 初始化BatchNorm权重
                init.constant_(m.bias, 0)  # 初始化偏置
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 初始化全连接层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的批量大小和通道数
        y = self.avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道注意力
        return x * y.expand_as(x)  # 通道注意力加权输入

# 高低频特征自适应增强模块
class HLFAE(nn.Module):  # HLFAE高低频特征自适应增强模块
    def __init__(self, dim, ratio=16):
        super(HLFAE, self).__init__()

        self.down = nn.AvgPool2d(kernel_size=2)  # 平均池化层，用于获取低频特征
        self.high_feature = feature_extraction(dim)  # 高频特征提取模块
        self.low = feature_extraction(dim)  # 低频特征提取模块
        self.conv1 = CBR(2 * dim, dim)  # 结合高频和低频特征的卷积模块

        # 用于计算低频特征的通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通过全连接层生成注意力权重
        self.fc1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid生成注意力权重
        self.att = SEAttention(dim)  # SE注意力模块

    def forward(self, x):
        # 第一步：分解高频和低频特征
        low = self.down(x)  # 低频特征通过池化获得
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 高频特征
        high = self.high_feature(high)  # 提取高频特征

        # 第二步：低频特征的增强
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(low))))  # 平均池化后经过FC生成注意力权重
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(low))))  # 最大池化后经过FC生成注意力权重
        low = low * self.sigmoid(avg_out + max_out)  # 低频特征加权
        low = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 恢复原始尺寸

        # 第三步：融合高低频特征
        out = torch.cat([high, low], dim=1)  # 高低频特征拼接
        out = self.att(self.conv1(out)) + x  # 使用SE注意力模块处理融合后的特征，并加回输入特征
        return out

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
class C2f_HLFAE(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(HLFAE(self.c) for _ in range(n))

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
        self.m = nn.Sequential(*(HLFAE(c_)  for _ in range(n)))

class C3k2_HLFAE(C2f_HLFAE):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else HLFAE(self.c) for _ in range(n)
        )

# 测试代码：输入BCHW，输出BCHW
if __name__ == '__main__':
    # 实例化模型对象
    model = HLFAE(dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)  # 随机生成尺寸为(1, 32, 64, 64)的输入
    # 执行前向传播
    output = model(input)
    # 打印输入和输出的尺寸
    print('input_size:', input.size())
    print('output_size:', output.size())
