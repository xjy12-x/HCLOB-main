import numbers

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import fft, nn


# 看Ai缝合怪b站视频：2025.8.30 更新的视频
class ComplexFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 保存原始数据类型
        original_dtype = x.dtype
        # 将输入转换为float32进行FFT
        x = x.to(torch.float32)
        x_fft = fft.fft2(x, dim=(-2, -1))
        real = x_fft.real.to(original_dtype)  # 转换回原始数据类型
        imag = x_fft.imag.to(original_dtype)
        return real, imag


class ComplexIFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, imag):
        # 将输入转换为float32
        original_dtype = real.dtype
        real = real.to(torch.float32)
        imag = imag.to(torch.float32)
        x_complex = torch.complex(real, imag)
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))
        return x_ifft.real.to(original_dtype)  # 转换回原始数据类型


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0, groups=in_channels * 2
        )

    def forward(self, x):
        return self.conv(x)


class Stage2_fft(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Multi-Scale Flow Gating Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )
        self.dwconv_2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=5, padding="same", groups=hidden_features, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.dwconv_2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class spr_sa(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim), nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x = x1 * x
        x = self.act(x)
        x = self.conv_1(x)
        return x


class AAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa = spr_sa(dim // 2, 2)
        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim // 2 * 3, dim // 2 * 3, kernel_size=3, stride=1, padding=1, groups=dim // 2 * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.0)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

        self.fft = Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1), nn.ReLU(), nn.Conv2d(dim // 4, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        b, _c, h, w = x.shape

        # 检查输入是否包含NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: AAFM输入包含NaN/Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        y, x_split = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        qkv = self.qkv_dwconv(self.qkv(x_split))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape

        # 计算gate_output
        gate_output = self.gate(x_split)

        # 检查gate_output是否包含NaN/Inf
        if torch.isnan(gate_output).any() or torch.isinf(gate_output).any():
            print("警告: gate_output包含NaN/Inf，使用默认值0.5")
            gate_mean = 0.5
        else:
            gate_mean = gate_output.view(b, -1).mean()
            # 确保gate_mean是有效的
            if torch.isnan(gate_mean) or torch.isinf(gate_mean):
                print("警告: gate_mean为NaN/Inf，使用默认值0.5")
                gate_mean = 0.5
            else:
                gate_mean = gate_mean.item()

        # 确保gate_mean在合理范围内
        gate_mean = max(0.1, min(0.9, gate_mean))

        # 计算dynamic_k
        dynamic_k = int(C * gate_mean)
        # 确保dynamic_k在合理范围内
        dynamic_k = max(1, min(C, dynamic_k))

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 限制注意力范围
        attn = torch.clamp(attn, -50.0, 50.0)

        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.0)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float("-inf")))

        attn = attn.softmax(dim=-1)

        # 防止softmax输出为NaN
        attn = torch.nan_to_num(attn, nan=0.0)

        out1 = attn @ v
        out2 = attn @ v
        out3 = attn @ v
        out4 = attn @ v

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        channel_map = self.channel_interaction(out_att)
        spatial_map = self.spatial_interaction(y_d)

        # 应用sigmoid前检查
        spatial_sig = torch.sigmoid(spatial_map)
        channel_sig = torch.sigmoid(channel_map)

        attened_x = out_att * spatial_sig
        conv_x = y_d * channel_sig

        x_combined = torch.cat([attened_x, conv_x], dim=1)
        out = self.project_out(x_combined)

        # stage 2
        out = self.fft(out)

        # 最终检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("警告: AAFM输出包含NaN/Inf，使用输入作为输出")
            out = x

        return out


# 输入 B C H W,  输出 B C H W
if __name__ == "__main__":
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(2, 32, 64, 64)
    # 创建 AAFM模块
    AAFM = AAFM(dim=32)
    # 将输入图像传入AAFM 模块进行处理
    output = AAFM(input)
    # 打印输入和输出的形状
    print("Ai缝合即插即用模块永久更新-AAFM_input_size:", input.size())
    print("Ai缝合即插即用模块永久更新-AAFM_output_size:", output.size())
    # ACM 2025 CCF-A 中的3个二次创新模块，CSAFM、GSCAFusion、LSCA在我的二次创新模块交流群
    # 适合冲SCI二、三区和四区，CCF-B/C,二次创新模块可以直接去发小论文！
    # 二次创新模块只更新在顶会顶刊二次创新交流，永久更新中->->
    # 二次创新改进商品链接在视频评论区，对二次创新模块感兴趣可以支持一下。
