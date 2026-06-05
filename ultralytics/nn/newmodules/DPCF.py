import torch
import torch.nn as nn
import torch.nn.functional as F
#https://arxiv.org/pdf/2505.23214
#看Ai缝合怪b站视频：2025.9.12 更新的视频
class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        # 定义可学习参数d，形状与p和i相同，这里假设p和i的形状为(batch_size, channel, w, h)
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))
    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        # 将self.d扩展为与p和i相同的形状
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class DPCF(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.ac = AdaptiveCombiner()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=in_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0))
         )

    def forward(self, x_low, x_high ):

        image_size = x_low.size(2)
        if x_low != None:
            x_low = torch.chunk(x_low, 4, dim=1)
        if x_high != None:
            x_high = F.interpolate(x_high, size=[image_size, image_size], mode='bilinear', align_corners=True)
            x_high = torch.chunk(x_high, 4, dim=1)
        x0 = self.ac(x_low[0], x_high[0])
        x1 = self.ac(x_low[1], x_high[1])
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 DPCF 模块
    DPCF = DPCF(in_features=32, out_features=32)
    # 将输入图像传入 DPCF 模块进行处理
    output = DPCF(input1,input2)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-DPCF_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-DPCF_output_size:', output.size())
    # AFEM自适应融合增强模块 是SCI一区2025 DPCF模块的二次创新模块，在二次创新交流群！
    # 二次创新改进交流群的模块，可以直接发论文
    # 二次创新群文件里面都是顶会顶刊论文模块的二次创新改进模块，可以直接发论文