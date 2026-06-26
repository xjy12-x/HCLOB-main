import torch.nn as nn


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # 确保mid_channels至少为1
        mid_channels = max(mid_channels, 1)

        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size, stride, padding, groups=max(1, mid_channels), bias=False
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class GBConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """GBConv模块 - 修复通道数问题版本 参数: in_channels: 输入通道数 out_channels: 输出通道数.
        """
        super().__init__()

        # 从args或kwargs中提取norm_type
        norm_type = kwargs.get("norm_type", "GN")

        print(f"GBConv调试: in_channels={in_channels}, out_channels={out_channels}, norm_type={norm_type}")

        # 安全计算中间通道数
        def safe_mid_channels(channels):
            return max(channels // 8, 1)

        # 如果输入输出通道数不同，需要残差连接适配
        self.need_shortcut = in_channels != out_channels
        if self.need_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.block1 = nn.Sequential(
            BottConv(in_channels, out_channels, safe_mid_channels(in_channels), 3, 1, 1),
            nn.BatchNorm2d(out_channels),  # 使用简单的BatchNorm避免复杂问题
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x

        # 如果需要适配残差连接的通道数
        if self.need_shortcut:
            residual = self.shortcut(residual)

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual


class GSConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """GSConv模块 - 修复通道数问题版本."""
        super().__init__()

        norm_type = kwargs.get("norm_type", "GN")
        print(f"GSConv调试: in_channels={in_channels}, out_channels={out_channels}, norm_type={norm_type}")

        def safe_mid_channels(channels):
            return max(channels // 8, 1)

        # 残差连接适配
        self.need_shortcut = in_channels != out_channels
        if self.need_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.block1 = nn.Sequential(
            BottConv(in_channels, out_channels, safe_mid_channels(in_channels), 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        self.block4 = nn.Sequential(
            BottConv(out_channels, out_channels, safe_mid_channels(out_channels), 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x

        if self.need_shortcut:
            residual = self.shortcut(residual)

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        gate = self.psi(x2)
        x = x1 * gate
        x = self.block4(x)

        return x + residual


# 注册模块
from mmcv.cnn import CONV_LAYERS

CONV_LAYERS.register_module(module=GBConv)
CONV_LAYERS.register_module(module=GSConv)
