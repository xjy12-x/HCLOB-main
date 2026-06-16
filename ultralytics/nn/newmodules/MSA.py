import torch
import torch.nn as nn


# https://arxiv.org/pdf/2506.20922
# ICCV 2025
class MSA(nn.Module):
    def __init__(self, in_channels, out_channels, levels):
        super().__init__()
        self.levels = levels
        # 下采样（Down）层：每个尺度级别应用卷积 -> ReLU -> 卷积
        self.downsample_layers = nn.ModuleList()
        for i in range(levels):
            # 下采样：卷积 -> ReLU -> 卷积
            i = i + 1
            if i == 1:
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, dilation=1, padding=1),  # C2D (3)
                        nn.ReLU(),
                        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),  # C2D (1)
                    )
                )
            else:
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2 ** (i - 1), stride=2 ** (i - 1)),
                        nn.Conv2d(
                            in_channels, in_channels, kernel_size=3, stride=1, dilation=2 * i + 1, padding=2 * i + 1
                        ),  # C2D (3)
                        nn.ReLU(),
                        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),  # C2D (1)
                    )
                )

        # 上采样（Up）层
        self.upsample_layers = nn.ModuleList()
        for i in range(levels):
            i = i + 1
            if i == 1:
                self.upsample_layers.append(nn.Identity())  # Up
            else:
                self.upsample_layers.append(nn.Upsample(scale_factor=2 ** (i - 1), mode="nearest"))  # Up

        # 最后的输出卷积层
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()
        self.rule = nn.ReLU()

    def Gate_Function(self, x):
        att = self.sigmod(self.conv1x1(x))
        output = att * x + (1 - att) * x
        return self.rule(self.conv1x1(output))

    def forward(self, x):
        # 存储每一层的特征
        features = []
        for i in range(self.levels):
            downsampled = self.downsample_layers[i](x)  # 下采样
            features.append(downsampled)

        # 多尺度特征融合
        output = features[0]
        output = self.Gate_Function(output)
        for i in range(1, self.levels):
            output += self.upsample_layers[i](self.Gate_Function(features[i]))  # 上采样并融合特征

        # 最后的卷积输出
        output = self.final_conv(output)

        return output


if __name__ == "__main__":
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 64, 256, 256)  # 输入大小为[1, 64, 256, 256]
    # 创建 MultiScaleAttention模块
    MSA = MSA(in_channels=64, out_channels=128, levels=4)
    # 将输入图像传入MultiScaleAttention模块进行处理
    output = MSA(input)
    # 打印输入和输出的形状
    print("Ai缝合即插即用模块永久更新_MSA_input_size:", input.size())
    print("Ai缝合即插即用模块永久更新_MSA_output_size:", output.size())
    # MSDAM 多尺度动态注意力模块 是ICCV 2025 MSA的顶会二次创新模块，在二次创新交流群！
    # 二次创新群文件里面都是顶会顶刊论文中的二次创新改进模块，可以直接去发小论文、完成毕业大论文！
