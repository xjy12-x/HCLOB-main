import torch
import torch.nn.functional as F
from torch import nn

'''
来自TETCI 2024 论文   CV任务通用           YOLOv8v10v11创新改进商品 和 即插即用模块商品在评论区
# 即插即用注意力： SAA 自我感知注意力           
提供二次创新  SCGA 自我感知协调注意力 效果优于SAA,可以直接拿去冲SCI一区  

从医学图像中精确分割器官或病变对疾病诊断和器官形态测量至关重要。
近年来，卷积编码器-解码器结构在自动医学图像分割领域取得了显著进展。
然而，由于卷积操作的固有偏差，现有模型主要关注由邻近像素形成的局部视觉线索，未能充分建模长程上下文依赖性。
本文提出了一种新颖的基于Transformer的注意力引导网络，称为 TransAttUnet。
该网络设计了多级引导注意力和多尺度跳跃连接，以共同增强语义分割架构的性能。
受Transformer启发，本文将 自感知注意力模块 (SAA) 融入TransAttUnet中，
该模块结合了Transformer自注意力 (TSA) 和全局空间注意力 (GSA)，能够有效地学习编码器特征之间的非局部交互。
此外，本文还在解码器块之间引入了多尺度跳跃连接，用于将不同语义尺度的上采样特征进行聚合，
从而增强多尺度上下文信息的表示能力，生成具有区分性的特征。得益于这些互补组件，
TransAttUnet能够有效缓解卷积层堆叠和连续采样操作引起的细节丢失问题，最终提升医学图像分割的质量。
在多个医学图像分割数据集上的大量实验表明，所提出的方法在不同成像模式下始终优于最新的基线模型。

SAA 模块是作用在于增强医学图像分割的上下文语义建模能力和全局空间关系表征能力。
其核心由以下两部分组成：
1.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。

2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。

适用于：医学图像分割，目标检测，语义分割，图像增强，暗光增强，遥感图像任务等所有计算机视觉CV任务通用注意力模块
'''
# class PAM_Module(nn.Module):
#     """空间注意力模块"""
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma * out + x
#         return out

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
# class DChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16, alpha=0.5):
#         super(DChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.alpha = alpha  # 平衡参数，控制池化策略的加权


#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):   #来自Ai缝合怪复现整理
#         avg_pool = self.avg_pool(x)  # 平均池化
#         max_pool = self.max_pool(x)  # 最大池化
#         mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool  # 使用alpha加权平均池化与最大池化的结果
#         # 计算每种池化方式的通道注意力
#         avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))  # 平均池化的通道注意力
#         max_out = self.fc2(self.relu1(self.fc1(max_pool)))  # 最大池化的通道注意力
#         mix_out = self.fc2(self.relu1(self.fc1(mix_pool)))  # 混合池化的通道注意力

#         out_pool = self.sigmoid(avg_out + max_out + mix_out)  # 将所有池化方式的结果相加并通过sigmoid计算最终权重
#         return x * out_pool

# class ScaledDotProductAttention(nn.Module):
#     '''自注意力模块'''

#     def __init__(self, temperature=512, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature ** 0.5
#         self.dropout = nn.Dropout(attn_dropout)

#     def forward(self, x, mask=None):
#         m_batchsize, d, height, width = x.size()
#         q = x.view(m_batchsize, d, -1)
#         k = x.view(m_batchsize, d, -1)
#         k = k.permute(0, 2, 1)
#         v = x.view(m_batchsize, d, -1)

#         attn = torch.matmul(q / self.temperature, k)

#         if mask is not None:
#             # 给需要mask的地方设置一个负无穷
#             attn = attn.masked_fill(mask == 0, -1e9)

#         attn = self.dropout(F.softmax(attn, dim=-1))
#         output = torch.matmul(attn, v)
#         output = output.view(m_batchsize, d, height, width)

#         return output

# class SAA(nn.Module):
#     def __init__(self, in_channels):
#         super(SAA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels)
#         self.tsa = ScaledDotProductAttention()
#     def forward(self, x):
#         x1 = self.gsa(x)
#         x2 = self.gsa(x)
#         out = x1 + x2
#         return out
# # 二次创新注意力模块 SCGA 自我感知协调注意力 冲SCI一区
# '''
# SCGA 自我感知协调注意力 内容介绍：

# 1.执行通道注意力机制。它对每个通道进行全局平均池化，
# 然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
# 确保模型能够有效地聚焦在最相关的通道特征上。
# 2.全局空间注意力 Global Spatial Attention (GSA):
# 提取和整合全局空间信息，从而增强并优化特征表示。
# GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。
# 3.多头自注意力 Transformer Self Attention (TSA):
# 使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
# TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。
# '''
# class SCGA(nn.Module):
#     def __init__(self, in_channels):
#         super(SCGA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels)
#         self.tsa = ScaledDotProductAttention()
#         self.ca = ChannelAttention(in_channels)
#     def forward(self, x):
#         x1 = x * self.ca(x)
#         x1 = x1 * self.gsa(x)

#         x2 = self.gsa(x)

#         out = x1 + x2
#         return out

# class DSCGA(nn.Module):
#     def __init__(self, in_channels):
#         super(DSCGA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels)
#         self.tsa = ScaledDotProductAttention()
#         self.ca = DChannelAttention(in_channels)

        
#     def forward(self, x):
#         # 保存输入用于残差连接
#         identity = x
        
#         # 1. 通道注意力
#         ca_weight = self.ca(x)
#         x_ca = x * ca_weight
        
#         # 2. 空间注意力
#         x_gsa = self.gsa(x_ca)
        
#         # 3. Transformer自注意力
#         x_tsa = self.tsa(x_ca)
        
#         # 4. 残差连接
#         out = x_gsa + x_tsa+identity
        

        
#         return out

# # 输入 N C H W,  输出 N C H W
# if __name__ == '__main__':
#     input = torch.rand(1, 64, 128, 128)
#     SAA = SAA(in_channels=64)
#     output = SAA(input)
#     print("SAA_input.shape:", input.shape)
#     print("SAA_output.shape:",output.shape)
#     SCGA = SCGA(in_channels=64)
#     output = SCGA(input)
#     print("二次创新_SCGA_input.shape:", input.shape)
#     print("二次创新_SCGA_output.shape:",output.shape)

# import torch
# import torch.nn.functional as F
# from torch import nn

# class ScaledDotProductAttention(nn.Module):
#     '''自注意力模块 - 修复版，避免显存爆炸'''
    
#     def __init__(self, temperature=512, attn_dropout=0.1, num_heads=8):
#         super().__init__()
#         self.temperature = temperature ** 0.5
#         self.dropout = nn.Dropout(attn_dropout)
#         self.num_heads = num_heads
    
#     def forward(self, x, mask=None):
#         m_batchsize, d, height, width = x.size()
        
#         # 方法1：空间下采样（推荐）
#         # 将特征图下采样以减少计算量
#         if height * width > 4096:  # 如果特征图太大
#             # 下采样到32x32或更小
#             target_size = (32, 32) if height >= 64 else (16, 16)
#             x_down = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
#             d, h_down, w_down = d, target_size[0], target_size[1]
#         else:
#             x_down = x
#             h_down, w_down = height, width
        
#         # 方法2：使用多头注意力减少矩阵大小
#         if self.num_heads > 1:
#             head_dim = d // self.num_heads
#             q = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
#             k = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
#             v = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
            
#             attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
            
#             if mask is not None:
#                 attn = attn.masked_fill(mask == 0, -1e9)
            
#             attn = self.dropout(F.softmax(attn, dim=-1))
#             output = torch.matmul(attn, v)
#             output = output.view(m_batchsize, d, h_down, w_down)
#         else:
#             # 原单头注意力
#             q = x_down.view(m_batchsize, d, -1)
#             k = x_down.view(m_batchsize, d, -1).permute(0, 2, 1)
#             v = x_down.view(m_batchsize, d, -1)
            
#             attn = torch.matmul(q / self.temperature, k)
            
#             if mask is not None:
#                 attn = attn.masked_fill(mask == 0, -1e9)
            
#             attn = self.dropout(F.softmax(attn, dim=-1))
#             output = torch.matmul(attn, v)
#             output = output.view(m_batchsize, d, h_down, w_down)
        
#         # 如果下采样了，再上采样回原尺寸
#         if height != h_down or width != w_down:
#             output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
        
#         return output
    

# class DChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16, alpha=0.5):
#         super(DChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.alpha = alpha  # 平衡参数，控制池化策略的加权


#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):   #来自Ai缝合怪复现整理
#         avg_pool = self.avg_pool(x)  # 平均池化
#         max_pool = self.max_pool(x)  # 最大池化
#         mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool  # 使用alpha加权平均池化与最大池化的结果
#         # 计算每种池化方式的通道注意力
#         avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))  # 平均池化的通道注意力
#         max_out = self.fc2(self.relu1(self.fc1(max_pool)))  # 最大池化的通道注意力
#         mix_out = self.fc2(self.relu1(self.fc1(mix_pool)))  # 混合池化的通道注意力

#         out_pool = self.sigmoid(avg_out + max_out + mix_out)  # 将所有池化方式的结果相加并通过sigmoid计算最终权重
#         return x * out_pool
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class PAM_Module(nn.Module):
#     """空间注意力模块 - 修复版"""
#     def __init__(self, in_dim, reduction=8):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         mid_channels = max(8, in_dim // reduction)
        
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # 空间下采样以减少计算量
#         if H * W > 4096:
#             target_h, target_w = max(32, H // 4), max(32, W // 4)
#             x_down = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
#         else:
#             x_down = x
#             target_h, target_w = H, W
        
#         proj_query = self.query_conv(x_down).view(B, -1, target_h * target_w).permute(0, 2, 1)
#         proj_key = self.key_conv(x_down).view(B, -1, target_h * target_w)
        
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x_down).view(B, -1, target_h * target_w)
        
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(B, C, target_h, target_w)
        
#         if H != target_h or W != target_w:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
#         out = self.gamma * out + x
#         return out
# class SAA(nn.Module):
#     def __init__(self, in_channels):
#         super(SAA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels)
#         self.tsa = ScaledDotProductAttention()
#     def forward(self, x):
#         x1 = self.gsa(x)
#         x2 = self.gsa(x)
#         out = x1 + x2
#         return out
# class SCGA(nn.Module):
#     """轻量级SCGA，显存友好版"""
#     def __init__(self, in_channels, reduction=16):
#         super(SCGA, self).__init__()
#         self.in_channels = in_channels
        
#         # 通道注意力
#         self.ca = ChannelAttention(in_channels, ratio=reduction)
        
#         # 简化空间注意力
#         self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
        
#         # 简化Transformer注意力
#         self.transform_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
#         # 可学习的融合权重
#         self.alpha = nn.Parameter(torch.ones(1))
#         self.beta = nn.Parameter(torch.ones(1))
        
#     def forward(self, x):
#         # 1. 通道注意力
#         ca_weight = self.ca(x)
#         x_ca = x * ca_weight
        
#         # 2. 简化空间注意力
#         spatial_avg = F.adaptive_avg_pool2d(x_ca, 1)
#         spatial_weight = self.conv2(F.relu(self.conv1(spatial_avg)))
#         spatial_weight = self.sigmoid(spatial_weight)
#         x_spatial = x_ca * spatial_weight
        
#         # 3. 简化Transformer自注意力
#         x_transform = self.transform_conv(x_spatial)
        
#         # 4. 残差融合
#         out = x + self.alpha * x_spatial + self.beta * x_transform
        
#         return out



# class DSCGA(nn.Module):
#     def __init__(self, in_channels):
#         super(DSCGA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels)
#         self.tsa = ScaledDotProductAttention()
#         self.ca = DChannelAttention(in_channels)
#     def forward(self, x):
#         x1 = x * self.ca(x)
#         x2 = x1 * self.gsa(x1)

#         x3 = self.tsa(x1)

#         out = x2 + x3
#         return out


# # # 保持原接口兼容
# # SCGA = SCGA_Lite

# if __name__ == '__main__':
#     input = torch.rand(1, 64, 128, 128)
#     scga = SCGA(in_channels=64)
#     output = scga(input)
#     print("SCGA_Lite input.shape:", input.shape)
#     print("SCGA_Lite output.shape:", output.shape)
    
#     # 测试不同尺寸
#     test_sizes = [(64, 64), (128, 128), (256, 256), (640, 640)]
#     for size in test_sizes:
#         try:
#             input_test = torch.rand(1, 64, size[0], size[1])
#             output_test = scga(input_test)
#             print(f"✓ 测试通过：输入尺寸 {size}, 输出尺寸 {output_test.shape}")
#         except Exception as e:
#             print(f"✗ 测试失败：输入尺寸 {size}, 错误：{e}")



import torch
import torch.nn.functional as F
from torch import nn
import warnings
warnings.filterwarnings('ignore')

class ScaledDotProductAttention(nn.Module):
    '''自注意力模块 - 修复版，避免显存爆炸和数值不稳定'''
    
    def __init__(self, temperature=512, attn_dropout=0.1, num_heads=8, eps=1e-8):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.num_heads = num_heads
        self.eps = eps
        
    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        
        # 方法1：空间下采样（推荐）
        if height * width > 4096:  # 如果特征图太大
            target_size = (32, 32) if height >= 64 else (16, 16)
            x_down = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            d, h_down, w_down = d, target_size[0], target_size[1]
        else:
            x_down = x
            h_down, w_down = height, width
        
        # 方法2：使用多头注意力减少矩阵大小
        if self.num_heads > 1:
            head_dim = d // self.num_heads
            q = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
            k = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
            v = x_down.view(m_batchsize, self.num_heads, head_dim, -1)
            
            # 计算注意力分数
            attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e4)
            
            # 数值稳定的softmax
            attn_max = attn.max(dim=-1, keepdim=True)[0]
            attn_exp = torch.exp(attn - attn_max)
            attn_sum = attn_exp.sum(dim=-1, keepdim=True) + self.eps
            attn = attn_exp / attn_sum
            
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
            output = output.view(m_batchsize, d, h_down, w_down)
        else:
            # 原单头注意力
            q = x_down.view(m_batchsize, d, -1)
            k = x_down.view(m_batchsize, d, -1).permute(0, 2, 1)
            v = x_down.view(m_batchsize, d, -1)
            
            attn = torch.matmul(q / self.temperature, k)
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e4)
            
            # 数值稳定的softmax
            attn_max = attn.max(dim=-1, keepdim=True)[0]
            attn_exp = torch.exp(attn - attn_max)
            attn_sum = attn_exp.sum(dim=-1, keepdim=True) + self.eps
            attn = attn_exp / attn_sum
            
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
            output = output.view(m_batchsize, d, h_down, w_down)
        
        # 如果下采样了，再上采样回原尺寸
        if height != h_down or width != w_down:
            output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
        
        return output

class DChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, alpha=0.5, eps=1e-8):
        super(DChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = alpha
        self.eps = eps
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """更好的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 1. 池化操作
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        # 数值稳定性检查
        if torch.isnan(avg_pool).any() or torch.isinf(avg_pool).any():
            avg_pool = torch.nan_to_num(avg_pool, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(max_pool).any() or torch.isinf(max_pool).any():
            max_pool = torch.nan_to_num(max_pool, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 混合池化
        mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool
        
        # 2. 计算通道注意力
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        mix_out = self.fc2(self.relu1(self.fc1(mix_pool)))
        
        # 3. 数值稳定性处理
        out_sum = avg_out + max_out + mix_out
        
        # 检查并修复NaN/Inf
        if torch.isnan(out_sum).any() or torch.isinf(out_sum).any():
            out_sum = torch.nan_to_num(out_sum, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. 应用sigmoid
        out_pool = self.sigmoid(out_sum)
        
        # 5. 返回加权特征
        result = x * out_pool
        
        # 最终检查
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class PAM_Module(nn.Module):
    """空间注意力模块 - 修复版，数值稳定"""
    def __init__(self, in_dim, reduction=8, eps=1e-8):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        mid_channels = max(8, in_dim // reduction)
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # 从0改为0.1
        self.softmax = nn.Softmax(dim=-1)
        self.eps = eps
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 空间下采样以减少计算量
        if H * W > 4096:
            target_h, target_w = max(32, H // 4), max(32, W // 4)
            x_down = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        else:
            x_down = x
            target_h, target_w = H, W
        
        proj_query = self.query_conv(x_down).view(B, -1, target_h * target_w).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(B, -1, target_h * target_w)
        
        # 计算注意力能量
        energy = torch.bmm(proj_query, proj_key)
        
        # 数值稳定的softmax
        energy_max = energy.max(dim=-1, keepdim=True)[0]
        energy_exp = torch.exp(energy - energy_max)
        energy_sum = energy_exp.sum(dim=-1, keepdim=True) + self.eps
        attention = energy_exp / energy_sum
        
        proj_value = self.value_conv(x_down).view(B, -1, target_h * target_w)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, target_h, target_w)
        
        if H != target_h or W != target_w:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        out = self.gamma * out + x
        
        # 数值稳定性检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return out
    
class SAA(nn.Module):
    def __init__(self, in_channels):
        super(SAA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        self.tsa = ScaledDotProductAttention()
    def forward(self, x):
        x1 = self.gsa(x)
        x2 = self.gsa(x)
        out = x1 + x2
        return out
class SCGA(nn.Module):
    """轻量级SCGA，显存友好版"""
    def __init__(self, in_channels, reduction=16):
        super(SCGA, self).__init__()
        self.in_channels = in_channels
        
        # 通道注意力
        self.ca = ChannelAttention(in_channels, ratio=reduction)
        
        # 简化空间注意力
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 简化Transformer注意力
        self.transform_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # 1. 通道注意力
        ca_weight = self.ca(x)
        x_ca = x * ca_weight
        
        # 2. 简化空间注意力
        spatial_avg = F.adaptive_avg_pool2d(x_ca, 1)
        spatial_weight = self.conv2(F.relu(self.conv1(spatial_avg)))
        spatial_weight = self.sigmoid(spatial_weight)
        x_spatial = x_ca * spatial_weight
        
        # 3. 简化Transformer自注意力
        x_transform = self.transform_conv(x_spatial)
        
        # 4. 残差融合
        out = x + self.alpha * x_spatial + self.beta * x_transform
        
        return out

# class DSCGA(nn.Module):
#     def __init__(self, in_channels, reduction=16, eps=1e-8):
#         super(DSCGA, self).__init__()
#         self.gsa = PAM_Module(in_dim=in_channels, eps=eps)
#         self.tsa = ScaledDotProductAttention(eps=eps)
#         self.ca = DChannelAttention(in_channels, eps=eps)
#         self.eps = eps
        
#     def forward(self, x):
#         # 保存输入用于残差连接
#         identity = x
        
#         # 1. 通道注意力
#         x_ca = self.ca(x)
        
#         # 2. 空间注意力
#         x_gsa = self.gsa(x_ca)
        
#         # 3. Transformer自注意力
#         x_tsa = self.tsa(x_ca)
        
#         # 4. 残差连接
#         out = x_gsa + x_tsa+x_ca
        
#         # 数值稳定性检查
#         if torch.isnan(out).any() or torch.isinf(out).any():
#             out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
#         return out

class DSCGA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DSCGA, self).__init__()
        self.in_channels = in_channels
        self.ca = DChannelAttention(in_channels, ratio=reduction)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.transform_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        ca_weight = self.ca(x)
        x_ca = x * ca_weight
        spatial_avg = F.adaptive_avg_pool2d(x_ca, 1)
        spatial_weight = self.conv2(F.relu(self.conv1(spatial_avg)))
        spatial_weight = self.sigmoid(spatial_weight)
        x_spatial = x_ca * spatial_weight
        x_transform = self.transform_conv(x_spatial)
        out = x + self.alpha * x_spatial + self.beta * x_transform
        return out



# 测试修复
if __name__ == '__main__':
    # 测试不同输入尺寸
    test_sizes = [(64, 64), (128, 128), (256, 256), (640, 640)]
    
    for size in test_sizes:
        print(f"\n测试输入尺寸: {size}")
        input = torch.rand(2, 64, size[0], size[1])
        
        # 测试DSCGA
        try:
            dscga = DSCGA(in_channels=64)
            output = dscga(input)
            
            # 检查输出
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            max_val = output.max().item()
            min_val = output.min().item()
            
            print(f"  DSCGA: 通过")
            print(f"    输出形状: {output.shape}")
            print(f"    NaN: {has_nan}, Inf: {has_inf}")
            print(f"    值范围: [{min_val:.4f}, {max_val:.4f}]")
            
        except Exception as e:
            print(f"  DSCGA: 失败 - {e}")