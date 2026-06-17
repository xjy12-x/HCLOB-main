import torch
import torch.nn as nn
from einops import rearrange


# 看Ai缝合怪b站视频：2025.9.8更新的视频
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            # nn.Dropout(0.1)
        )


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super().__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            SeparableConvBNReLU(F1, F2, kernel_size, dilation=rate),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = SeparableConvBNReLU(dim * 2, (dim // 2) * 3, 3, 1, 1)
        self.att_dim = dim // 2

    def forward(self, x, y):
        b, _c, h, w = x.shape
        fm = self.conv(torch.concat([x, y], dim=1))

        Q, K, V = rearrange(fm, "b (qkv c) h w -> qkv b h c w", qkv=3, b=b, c=self.att_dim, h=h, w=w)

        dots = Q @ K.transpose(-2, -1)
        attn = dots.softmax(dim=-1)
        attn = attn @ V
        attn = attn.view(b, -1, h, w)

        return attn


class MSAM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.branch1 = nn.Sequential(
            SeparableConvBNReLU(dim_in[0], dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            SeparableConvBNReLU(dim_in[1], dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2),
        )
        self.branch3 = nn.Sequential(SeparableConvBNReLU(dim_in[2], dim_out, kernel_size=3, stride=2))
        self.branch4 = nn.Sequential(nn.Conv2d(dim_in[3], dim_out, kernel_size=1), nn.BatchNorm2d(dim_out), nn.ReLU6())
        self.merge = nn.Sequential(nn.Conv2d(4 * dim_out, dim_out, kernel_size=1), nn.BatchNorm2d(dim_out), nn.ReLU6())
        self.resblock = nn.Sequential(
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
        )
        self.transformer = SelfAttention(dim_out)
        self.conv = nn.Conv2d(dim_out // 2 * 10, dim_out, 1)
        self.dim_out = dim_out

    def forward(self, x, skip_list):
        b, _c, h, w = x.shape
        list1 = []
        list2 = []

        x1 = self.branch1(skip_list[0])
        x2 = self.branch2(skip_list[1])
        x3 = self.branch3(skip_list[2])
        x = self.branch4(x)

        # CNN
        merge = self.merge(torch.cat([x, x1, x2, x3], dim=1))
        merge = self.resblock(merge)

        # Transformer
        list1.append(x)
        list1.append(x3)
        list1.append(x2)
        list1.append(x1)

        for i in range(len(list1)):
            for j in range(len(list1)):
                if i <= j:
                    att = self.transformer(list1[i], list1[j])
                    list2.append(att)

        for j in range(len(list2)):
            list2[j] = list2[j].view(b, self.dim_out // 2, h, w)

        out = self.conv(torch.concat(list2, dim=1))

        return out + merge


if __name__ == "__main__":
    # 设置输入的 batch size 和空间尺寸
    B, H, W = 1, 16, 16  # 空间尺寸要和 MSAM 中下采样后的分支相匹配
    dim_out = 128  # MSAM 输出通道数
    # 模拟4个分支输入，按 MSAM 模块需要的维度创建
    skip_1 = torch.randn(B, 64, H * 8, W * 8)  # 分支1输入
    skip_2 = torch.randn(B, 128, H * 4, W * 4)  # 分支2输入
    skip_3 = torch.randn(B, 256, H * 2, W * 2)  # 分支3输入
    x = torch.randn(B, 512, H, W)  # 主输入 (传入branch4)

    # 构造 skip list
    skip_list = [skip_1, skip_2, skip_3]

    # 初始化 MSAM 模块
    msam = MSAM(dim_in=[64, 128, 256, 512], dim_out=512)
    # 前向传播
    output = msam(x, skip_list)
    # 打印输出形状
    print("Ai缝合即插即用模块永久更新-MSAM 输入张量形状:", x.shape)
    print("Ai缝合即插即用模块永久更新-MSAM 输出张量形状:", output.shape)
    # MPSAM 多极稀疏注意力融合模块 是TGRS 2025 MSAM的顶刊二次创新模块，在二次创新交流群！
    # 二次创新群文件里面都是顶会顶刊论文中的二次创新改进模块，可以直接去发小论文、完成毕业大论文！
