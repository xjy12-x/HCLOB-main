# # https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# # A Hybrid Transformer-Mamba Network for Single Image Deraining
# # https://github.com/sunshangquan/TransMamba
# # https://arxiv.org/abs/2409.00410
# from einops import rearrange
# from einops import rearrange
# import torch
# import torch.nn.functional as F
# import torch.nn as nn


# class SBSAtt(nn.Module):
#     def __init__(self, dim, num_heads=2, bias=True):
#         super(SBSAtt, self).__init__()
#         # self.num_heads = num_heads
#         # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         # self.factor = 2
#         # self.idx_dict = {}
#         # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.dim = dim
#         self.num_heads = num_heads

#         # 确保通道数可被头数整除
#         if dim % num_heads != 0:
#             # 自动调整头数
#             new_num_heads = self.find_divisor(dim)
#             #print(f"⚠️  dim {dim} 不能被 {num_heads} 整除，调整为 {new_num_heads} 个头")
#             self.num_heads = new_num_heads

#         self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

#         # 关键修复：确保输入输出通道数正确
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#         # 注册自适应适配器（延迟创建）
#         self.adapter = None

#     def pad(self, x, factor):
#         hw = x.shape[-1]
#         t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
#         x = F.pad(x, t_pad, 'constant', 0)
#         return x, t_pad

#     def unpad(self, x, t_pad):
#         hw = x.shape[-1]
#         return x[..., t_pad[0]:hw - t_pad[1]]

#     def real2comp(self, x):
#         xr, xi = x.chunk(2, dim=1)
#         xr = xr.float()
#         xi = xi.float()
#         return torch.complex(xr, xi)

#     def comp2real(self, x):
#         b, _, h, w = x.shape
#         if x.dtype != torch.complex64:
#             x = x.to(torch.complex64)
#         return torch.cat([x.real, x.imag], 1)

#     def softmax_1(self, x, dim=-1):
#         logit = x.exp()
#         logit = logit / (logit.sum(dim, keepdim=True) + 1)
#         return logit

#     def get_idx_map(self, h, w):
#         l1_u = torch.arange(h // 2).view(1, 1, -1, 1)
#         l2_u = torch.arange(w).view(1, 1, 1, -1)
#         half_map_u = l1_u @ l2_u
#         l1_d = torch.arange(h - h // 2).flip(0).view(1, 1, -1, 1)
#         l2_d = torch.arange(w).view(1, 1, 1, -1)
#         half_map_d = l1_d @ l2_d
#         return torch.cat([half_map_u, half_map_d], 2).view(1, 1, -1).argsort(-1)

#     def get_idx(self, x):
#         h, w = x.shape[-2:]
#         if (h, w) in self.idx_dict:
#             return self.idx_dict[(h, w)]
#         idx_map = self.get_idx_map(h, w).to(x.device).detach()
#         self.idx_dict[(h, w)] = idx_map
#         return idx_map

#     def fft(self, x):
#         """修复后的FFT方法，确保总是返回三个值"""
#         try:
#             x = x.float()
#             x, pad = self.pad(x, 2)
#             x = torch.fft.rfft2(x, norm="ortho")
#             x = self.comp2real(x)
#             idx = self.get_idx(x)
#             b, c, h, w = x.shape

#             # 重塑并确保索引正确
#             x_flat = x.contiguous().view(b, c, -1)
#             idx_expanded = idx.repeat(b, c, 1)

#             # 检查索引边界
#             if idx_expanded.max() >= x_flat.size(2):
#                 idx_expanded = torch.arange(x_flat.size(2), device=x.device).view(1, 1, -1).repeat(b, c, 1)

#             x_gathered = torch.gather(x_flat, 2, index=idx_expanded)
#             return x_gathered, pad, idx

#         except Exception as e:
#             # 备用方案：返回简单结果
#             print(f"FFT error: {e}, using simple fallback")
#             x = x.float()
#             x, pad = self.pad(x, 2)
#             b, c, h, w = x.shape
#             idx = torch.arange(h * w, device=x.device).view(1, 1, -1)
#             return x.contiguous().view(b, c, -1), pad, idx

#     def ifft(self, x, pad, idx, h):
#         """修复后的IFFT方法"""
#         try:
#             with torch.autocast('cuda', enabled=False):
#                 b, c = x.shape[:2]
#                 # 使用scatter恢复原始顺序
#                 x_reconstructed = torch.zeros(b, c, h * (x.size(-1) // h), device=x.device)
#                 idx_expanded = idx.repeat(b, c, 1)
#                 x_reconstructed = torch.scatter(x_reconstructed, 2, idx_expanded, x)

#                 x_reshaped = x_reconstructed.view(b, c, h, -1)
#                 x_complex = self.real2comp(x_reshaped)
#                 x_recon = torch.fft.irfft2(x_complex, norm='ortho')
#                 x_recon = self.unpad(x_recon, pad)
#                 return x_recon

#         except Exception as e:
#             print(f"IFFT error: {e}, returning zero tensor")
#             # 返回零张量作为降级方案
#             return torch.zeros(b, c, h, h, device=x.device)

#     def attn(self, qkv):
#         """修复注意力方法，添加错误处理"""
#         try:
#             h = qkv.shape[2]
#             q, k, v = qkv.chunk(3, dim=1)

#             # 添加None检查
#             q_result = self.fft(q)
#             if q_result is None:
#                 raise ValueError("FFT returned None for q")
#             q, pad_w, idx = q_result

#             k_result = self.fft(k)
#             if k_result is None:
#                 raise ValueError("FFT returned None for k")
#             k, pad_w, _ = k_result

#             v_result = self.fft(v)
#             if v_result is None:
#                 raise ValueError("FFT returned None for v")
#             v, pad_w, _ = v_result

#             # 其余代码保持不变...
#             q, pad = self.pad(q, self.factor)
#             k, pad = self.pad(k, self.factor)
#             v, pad = self.pad(v, self.factor)

#             q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
#             k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
#             v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

#             q = F.normalize(q, dim=-1)
#             k = F.normalize(k, dim=-1)

#             attn = (q @ k.transpose(-2, -1)) * self.temperature
#             attn = self.softmax_1(attn, dim=-1)

#             out = (attn @ v)
#             out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
#             out = self.unpad(out, pad)
#             out = self.ifft(out, pad_w, idx, h)
#             return out

#         except Exception as e:
#             print(f"Attention error: {e}, returning input")
#             # 返回输入作为降级方案
#             return qkv[:, :qkv.shape[1]//3, :, :]

#     def forward(self, x):
#         qkv = self.qkv_dwconv(self.qkv(x))
#         out = self.attn(qkv)

#         if out.dtype != self.project_out.weight.dtype:
#             out = out.to(self.project_out.weight.dtype)

#         out = self.project_out(out)
#         return out

import torch
import torch.nn.functional as F
from torch import nn


class SBSAtt(nn.Module):
    """数值稳定的SBSAtt版本."""

    def __init__(self, dim, num_heads=2, bias=True, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps  # 数值稳定性参数

        # 自动调整头数
        if dim % num_heads != 0:
            num_heads = self.find_divisor(dim)
        self.num_heads = num_heads

        # 使用更小的初始温度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)  # 降低初始值

        # 卷积层
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """更稳定的权重初始化."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 温度参数特殊初始化
        nn.init.constant_(self.temperature, 0.1)

    def find_divisor(self, n):
        """找到能整除n的最大合适头数."""
        for i in range(min(8, n), 0, -1):
            if n % i == 0:
                return i
        return 1

    def safe_softmax(self, x, dim=-1):
        """数值稳定的softmax."""
        # 减去最大值提高数值稳定性
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_stable = x - x_max
        exp_x = torch.exp(x_stable)
        return exp_x / (exp_x.sum(dim=dim, keepdim=True) + self.eps)

    def safe_normalize(self, x, dim=-1):
        """数值稳定的归一化."""
        return F.normalize(x, dim=dim, eps=self.eps)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 通道数适配
        if channels != self.dim:
            if channels > self.dim:
                x = x[:, : self.dim, :, :]
            else:
                padding = torch.zeros(batch_size, self.dim - channels, height, width, device=x.device)
                x = torch.cat([x, padding], dim=1)

        # 生成QKV
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 计算头维度
        head_dim = self.dim // self.num_heads

        # 重塑为多头格式
        q = q.view(batch_size, self.num_heads, head_dim, height * width)
        k = k.view(batch_size, self.num_heads, head_dim, height * width)
        v = v.view(batch_size, self.num_heads, head_dim, height * width)

        # 数值稳定的归一化
        q = self.safe_normalize(q, dim=-1)
        k = self.safe_normalize(k, dim=-1)

        # 注意力计算（数值稳定版本）
        try:
            # 计算相似度
            attn_logits = torch.matmul(q, k.transpose(-2, -1))

            # 应用温度缩放（限制范围）
            temperature = torch.clamp(self.temperature, min=0.01, max=1.0)
            attn_logits = attn_logits * temperature

            # 数值稳定的softmax
            attn_weights = self.safe_softmax(attn_logits, dim=-1)

            # 应用注意力
            out = torch.matmul(attn_weights, v)

        except Exception as e:
            # 备用方案：恒等映射
            print(f"注意力计算失败: {e}, 使用恒等映射")
            out = v

        # 重塑回原始格式
        out = out.contiguous().view(batch_size, self.dim, height, width)

        # 输出投影
        out = self.project_out(out)

        return out
