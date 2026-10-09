from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# 代码地址：https://github.com/saadwazir/MCADS-Decoder/blob/main/mcadsDecoder.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

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


# DSUB: Depth-to-Space Upsampling Block
class DSUB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale

        # pre-up: Conv3x3 + ReLU，把通道扩到 out_ch * r^2 以便 PixelShuffle
        self.pre_conv = nn.Conv2d(in_ch, out_ch * (scale**2), kernel_size=3, padding=1, bias=True)
        self.pre_act = nn.ReLU(inplace=True)

        # D2S（PixelShuffle）
        self.ps = nn.PixelShuffle(scale)

        # post-up: Conv3x3 + ReLU（轻精炼）
        self.post_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
        self.post_act = nn.ReLU(inplace=True)

        # 最终用 CB 进一步精炼（与论文 Fig.1(d) 收尾一致）
        self.refine = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = self.ps(x)  # H,W 放大 scale 倍，通道缩到 out_ch
        x = self.post_conv(x)
        x = self.post_act(x)
        x = self.refine(x)
        return x


# ---------------------------
# EUB: Effective Upsampling Block
# ---------------------------
class EUB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        # 第一段 CB：in_ch → out_ch
        self.cb1 = ConvBlock(in_ch, out_ch)
        # 上采样（双线性）
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        # 第二段 CB：out_ch → out_ch
        self.cb2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.cb1(x)
        x = self.upsample(x)
        x = self.cb2(x)
        return x


# Residual Block (RB)
class ResidualRB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.conv3(x)
        y = self.conv1(y)
        y = x + y
        y = self.bn(y)
        y = self.act(y)
        return y


class RLAB(nn.Module):
    """Residual Linear Attention Block - n_rb: RB 迭代次数，建议各stage从深到浅用 [5,4,3,2,1] - proj_dim: 注意力中的 dk/dv（不设即用通道数）.
    """

    def __init__(self, in_ch_skip: int, n_rb: int = 3, proj_dim: int | None = None):
        super().__init__()
        self.n_rb = n_rb
        # 1) 对 skip 分支做多次 RB 精炼（式(8)）
        self.rb_blocks = nn.ModuleList([ResidualRB(in_ch_skip) for _ in range(n_rb)])
        # 2) 先用 CB 精炼（与论文“Following the concatenation, the combined feature map is refined”一致）
        out_ch = in_ch_skip
        in_ch_dec = 2 * in_ch_skip
        self.refine = ConvBlock(2 * in_ch_skip, out_ch)

        # 3) 线性注意力：1x1投影成Q/K/V；把(H,W)展平为N
        d_model = out_ch if proj_dim is None else proj_dim
        self.q = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.k = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.v = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.out = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=False)  # 回到out_ch

        self.scale = d_model**0.5  # sqrt(dk)

        self.eub = EUB(in_ch_skip, in_ch_skip)
        self.dsub = DSUB(in_ch_dec, in_ch_skip)

    def forward(self, skip_feat):
        # RB refinement on skip
        x_rb = skip_feat
        for rb in self.rb_blocks:
            x_rb = rb(x_rb)  # 多次迭代，深层更多

        dec_feat = skip_feat  # 保证skip_feat和dec_feat 特征 通道数相同

        if dec_feat.shape[2] != skip_feat.shape[2]:  # 保证skip_feat和dec_feat 特征 尺寸大小相同
            dec_feat = self.eub(dec_feat)

        # 融合
        x_1 = torch.cat([x_rb, dec_feat], dim=1)  # (B, 2*C_skip, H, W)

        x_ref = self.refine(x_1)  # (B, out_ch, H, W)

        # 线性注意力：把空间展平成 token 维
        B, _C, H, W = x_ref.shape
        H * W

        q = self.q(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)
        k = self.k(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)
        v = self.v(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)

        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)  # (B, N, N)  —— 式(10)

        y = torch.matmul(attn, v)  # (B, N, d)
        y = y.transpose(1, 2).reshape(B, -1, H, W)  # (B, d, H, W)
        y = self.out(y)  # (B, out_ch, H, W)

        out = y + self._proj_if_needed(x_1, y.shape[1])  # 更稳的做法：显式投影
        return out

    # 为了稳妥，给残差分支加一个自动投影（如果通道不匹配）
    def _proj_if_needed(self, x, out_ch):
        if x.shape[1] == out_ch:
            return x
        if not hasattr(self, "proj_res"):
            self.proj_res = nn.Conv2d(x.shape[1], out_ch, 1, bias=False)
        return self.proj_res(x)


# 输入 B H W C , 输出 B  H W C
if __name__ == "__main__":
    print("第一种情况：尺寸和通道数都不相同进行特征融合")
    input_dec = torch.randn(1, 64, 32, 32)
    input_skip = torch.randn(1, 32, 64, 64)

    RLAB1 = RLAB(in_ch_skip=32, in_ch_dec=64, n_rb=3)  # 创建  RLAB 模块实例
    output = RLAB1(input_skip, input_dec)
    print("RLAB1_输入input_dec张量形状:", input_dec.shape)
    print("RLAB1_输入input_skip张量形状:", input_skip.shape)
    print("RLAB1_输出张量形状:", output.shape)

    print("\n第二种情况：尺寸和通道数都相同进行特征融合")
    input_dec2 = torch.randn(1, 64, 64, 64)
    input_skip2 = torch.randn(1, 64, 64, 64)
    RLAB2 = RLAB(in_ch_skip=64, in_ch_dec=64, n_rb=3)  # 创建  RLAB 模块实例
    output = RLAB2(input_skip2, input_dec2)
    print("RLAB2_输入input_dec2张量形状:", input_dec2.shape)
    print("RLAB2_输入input_skip2张量形状:", input_skip2.shape)
    print("RLAB2_输出张量形状:", output.shape)

    print("有关RLAB的二次创新，会更新在顶会顶刊二次创新改进交流群！，二次创新改进交流群会持续更新中")
    # MALFM是CVPR2025 RLAB的二次创新模块，在二次创新交流群，二次创新群文件里面都是顶会顶刊论文模块的二次创新改进模块，可以直接发论文
