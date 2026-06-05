import torch
import torch.nn as nn
#代码地址：https://github.com/saadwazir/MCADS-Decoder/blob/main/mcadsDecoder.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = avg_out + max_out
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # Input will have 4 channels: mean, max, min, sum
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, groups=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        sum_out = torch.sum(x, dim=1, keepdim=True)

        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        attention = self.conv(pool)
        return x * attention


class CASAB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CASAB, self).__init__()
        self.convblock = ConvBlock(in_channels,in_channels)
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention() # 主要是对空间注意力模块做了二次创新

    def forward(self, x):
        x = self.convblock(x)
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return ca + sa

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    CASAB= CASAB(in_channels=64)
    output =  CASAB(input)
    print('Ai缝合即插即用模块永久更新-CASAB input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-CASAB output_size:', output.size())

    print('有关CASAB的二次创新，会更新在顶会顶刊二次创新改进交流群！，二次创新改进交流群会持续更新中')
    # DSCAM是CVPR2025 CASAB的二次创新模块，二次创新群文件里面都是顶会顶刊论文模块的二次创新改进模块，可以直接发论文