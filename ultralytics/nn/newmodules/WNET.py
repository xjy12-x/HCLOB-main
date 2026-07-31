import torch
from timm.models.layers import DropPath
from torch import nn

BNNorm2d = nn.BatchNorm2d
LNNorm = nn.LayerNorm
Activation = nn.GELU


# CVPR 2025 医学图像分割
##看Ai缝合怪b站视频：2025.6.23更新的视频
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation(),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False), BNNorm2d(ch_out), Activation()
        )

    def forward(self, x):
        x = self.down(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn2 = BNNorm2d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=1), BNNorm2d(planes))

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        if self.inplanes != self.planes:
            identity = self.down(x)

        out = self.bn2(out) + identity
        out = self.act(out)

        return out


class LSB_down_or_up(nn.Module):
    def __init__(self, c1, c2, down_or_up=None, groups=1):
        """修改参数顺序以匹配YOLO的调用方式.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            down_or_up: 'down'表示下采样，'up'表示上采样，None表示不改变尺寸
            groups: 分组卷积的组数
        """
        super().__init__()

        # 重新计算隐藏层维度
        hidden_planes = 2 * c1

        # 构建基本块
        if down_or_up is None or down_or_up == "" or down_or_up == "none":
            # 如果不指定上下采样，只做特征变换
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=c1, planes=hidden_planes, groups=groups),
                nn.Conv2d(hidden_planes, c2, 1, 1, 0, bias=False),
                BNNorm2d(c2),
                Activation(),
            )

        elif str(down_or_up).lower() == "down":
            # 下采样
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=c1, planes=hidden_planes, groups=groups), down_conv(hidden_planes, c2)
            )

        elif str(down_or_up).lower() == "up":
            # 上采样
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=c1, planes=hidden_planes, groups=groups), up_conv(hidden_planes, c2)
            )

        else:
            # 默认情况
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=c1, planes=hidden_planes, groups=groups),
                nn.Conv2d(hidden_planes, c2, 1, 1, 0, bias=False),
                BNNorm2d(c2),
                Activation(),
            )

    def forward(self, x):
        return self.BasicBlock(x)


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GSB(nn.Module):
    def __init__(self, in_dim, out_dim, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0, sr_ratio=1):
        super().__init__()

        self.in_dim = in_dim
        self.dim = out_dim

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(out_dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = GroupNorm(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, out_features=out_dim, drop=drop)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 输入 B C H W,  输出 B C H W
if __name__ == "__main__":
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建 GSB 模块
    GSB = GSB(in_dim=32, out_dim=32)
    # 将输入图像传入GSB 模块进行处理
    output = GSB(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print("Ai缝合即插即用模块永久更新-GSB_input_size:", input.size())
    print("Ai缝合即插即用模块永久更新-GSB_output_size:", output.size())

    # 创建 LSB 模块
    LSB_up = LSB_down_or_up(inplanes=32, outplanes=32, down_or_up="up")
    # 将输入图像传入LSB_up 模块进行上采样处理
    output = LSB_up(input)
    # 打印输入和输出的形状
    print("Ai缝合即插即用模块永久更新-LSB_up_input_size:", input.size())
    print("Ai缝合即插即用模块永久更新-LSB_up_output_size:", output.size())

    # 创建 LSB 模块
    LSB_down = LSB_down_or_up(inplanes=32, outplanes=32, down_or_up="down")
    # 将输入图像传入LSB_down模块进行下采样处理
    output = LSB_down(input)
    # 打印输入和输出的形状
    print("Ai缝合即插即用模块永久更新-LSB_down_input_size:", input.size())
    print("Ai缝合即插即用模块永久更新-LSB_down_output_size:", output.size())
