import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3
#看Ai缝合怪b站视频：2025.8.2 更新的视频

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
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   #来自Ai缝合怪复现整理
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
    
class DChannelAttention(nn.Module):
    def __init__(self, in_planes=32, alpha=0.5):
        super(DChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = alpha  # 平衡参数，控制池化策略的加权


        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   #来自Ai缝合怪复现整理
        avg_pool = self.avg_pool(x)  # 平均池化
        max_pool = self.max_pool(x)  # 最大池化
        mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool  # 使用alpha加权平均池化与最大池化的结果
        # 计算每种池化方式的通道注意力
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))  # 平均池化的通道注意力
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))  # 最大池化的通道注意力
        mix_out = self.fc2(self.relu1(self.fc1(mix_pool)))  # 混合池化的通道注意力

        out_pool = self.sigmoid(avg_out + max_out + mix_out)  # 将所有池化方式的结果相加并通过sigmoid计算最终权重
        return x * out_pool


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size   #来自Ai缝合怪复现整理
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=1)  # [B, k*k, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   #来自Ai缝合怪复现整理
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(   #来自Ai缝合怪复现整理
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att


# Residual Channel Spatial Attention Block (RCSAB)
class RCSAB(nn.Module):
    def __init__(self,n_feat, bn=True, act=nn.ReLU(True)):

        super(RCSAB, self).__init__()   #来自Ai缝合怪复现整理
        modules_body = []
        
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3,padding=1, bias=False))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(in_planes=n_feat))
        modules_body.append(DynamicSpatialAttention(in_channels=n_feat))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class DRCSAB(nn.Module):
    def __init__(self,n_feat, bn=True, act=nn.ReLU(True)):

        super(DRCSAB, self).__init__()   #来自Ai缝合怪复现整理
        modules_body = []
        
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3,padding=1, bias=False))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(DChannelAttention(in_planes=n_feat))
        modules_body.append(DynamicSpatialAttention(in_channels=n_feat))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class CRCSAB(nn.Module):
    def __init__(self,n_feat, bn=True, act=nn.ReLU(True)):

        super(CRCSAB, self).__init__()   #来自Ai缝合怪复现整理
        in_channels=n_feat
        self.in_channels=in_channels
        modules_body = []    
    # 定义深度可分离卷积（DWConv）模块，用于多尺度卷积
        self.dwconvs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            # 3x3 深度可分离卷积
            nn.BatchNorm2d(self.in_channels),  # 批量归一化
            nn.ReLU6(inplace=True))  # ReLU6 激活函数

        # 定义 1x1 卷积，用于通道融合
        self.conv1X1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义不同尺度的深度可分离卷积（3x3, 5x5, 7x7, 9x9）
        self.dwconv3X3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv5X5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 5 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv7X7 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 7, 1, 7 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv9X9 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 9, 1, 9 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3,padding=1, bias=False))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(in_planes=n_feat))
        modules_body.append(DynamicSpatialAttention(in_channels=n_feat))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        # 多尺度卷积
        x = self.dwconv3X3(x)  # 3x3 卷积
        x = x + self.dwconv5X5(x) + self.dwconv7X7(x) + self.dwconv9X9(x)  # 加上其他尺度的卷积输出
        x = self.conv1X1(x)  # 1x1 卷积，调整通道数
        res = self.body(x)
        res += x
        return res
    

    
class HLRCSAB(nn.Module):
    def __init__(self,n_feat, bn=True, act=nn.ReLU(True)):

        super(HLRCSAB, self).__init__()   #来自Ai缝合怪复现整理
        modules_body = []
        self.down = nn.AvgPool2d(kernel_size=2)  # 平均池化层，用于获取低频特征
        self.high_feature = feature_extraction(n_feat)  # 高频特征提取模块
        self.conv1 = CBR(2 * n_feat, n_feat)  # 结合高频和低频特征的卷积模块

        modules_body.append(ChannelAttention(in_planes=n_feat))
        modules_body.append(DynamicSpatialAttention(in_channels=n_feat))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        low = self.down(x)  # 低频特征通过池化获得
        low = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 恢复原始尺寸
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 高频特征
        high = self.high_feature(high)  # 提取高频特征
        x_cat = torch.cat([high, low], dim=1)  # 高低频特征拼接
        x_map = self.conv1(x_cat)  # [B, C, H, W]

        res = self.body(x_map)
        res += x
        return res
    # def forward(self, x):
    #     # 1. 频域解耦
    #     low = self.down(x)  # [B, C, H/2, W/2]
    #     high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)  # [B, C, H, W]
        
    #     # 2. 双分支多尺度提取（修复逻辑漏洞1：让 self.low 也起作用，否则定义了却没用）
    #     high = self.high_feature(high)  # [B, C, H, W]

        
    #     # 3. 上采样对齐并拼接（修复致命Bug：必须把 low 上采样回 H, W 才能和 high 拼接）
    #     low_up = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True) # [B, C, H, W]
    #     x_cat = torch.cat([high, low_up], dim=1)  # [B, 2C, H, W]
        
    #     # 4. 跨维度降维映射（修复逻辑漏洞2：调用 self.conv1 将 2C 降回 C，否则下一步会报错通道不匹配）
    #     x_map = self.conv1(x_cat)  # [B, C, H, W]
        
    #     # 5. 注意力净化与残差连接（注意：残差基准应该是降维后的 x_map，而不是拼接前的 2C 特征）
    #     res = self.body(x_map)
    #     res += x_map  
    #     return res

    
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

class Bottleneck_RCSAB(nn.Module):
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
        self.Attention = RCSAB(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_RCSAB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_RCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
        self.m = nn.Sequential(*(Bottleneck_RCSAB(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_RCSAB(C2f_RCSAB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
class Bottleneck_HLRCSAB(nn.Module):
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
        self.Attention =CRCSAB(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_HLRCSAB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_HLRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

class C3k2_HLRCSAB(C2f_HLRCSAB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_HLRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

class Bottleneck_CRCSAB(nn.Module):
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
        self.Attention = CRCSAB(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_CRCSAB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

class C3k2_CRCSAB(C2f_CRCSAB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

class Bottleneck_DRCSAB(nn.Module):
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
        self.Attention = DRCSAB(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_DRCSAB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

class C3k2_DRCSAB(C2f_DRCSAB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DRCSAB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

class Bottleneck_DSA(nn.Module):
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
        self.Attention = DynamicSpatialAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_DSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
        self.m = nn.Sequential(*(Bottleneck_DSA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_DSA(C2f_DSA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# 创建一个RCSAB实例
if __name__ == "__main__":
    # 设置随机输入 input 特征图 B C H W
    input = torch.randn(1,64,128,128)
    RCSAB = RCSAB(n_feat=64)
    # DSA =DynamicSpatialAttention(in_channels=64)
    output=RCSAB(input)
    print("Ai缝合怪整理的RCSAB_输入张量形状:", input.shape)
    print("Ai缝合怪整理的RCSAB_输出张量形状:", output.shape)
    print("Ai缝合怪DPLKA二次创新改进模块，只更新在顶会顶刊二次创新模块交流群！")
    #TGRS2025 RCSAB模块的二次创新，DPLKA在我的二次创新模块改进交流群，可以直接去发小论文！
    #DPLKA二次创新模块只更新二次创新交流，永久更新中
    #二次创新改进商品链接在视频评论区
