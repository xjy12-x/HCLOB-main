import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
from torch import Tensor

"""
来自Arxiv 2025最新论文  LWGA 轻量级分组注意力模块

遥感（RS）视觉任务在学术研究和实际应用中具有重要意义。然而，这些任务面临着多个挑战，尤其是在单幅图像中检测和识别尺度变化显著的多个目标，
这严重影响了特征提取的有效性。尽管以往的双分支或多分支架构在一定程度上缓解了尺度变化的问题，但也显著增加了计算开销和参数量，
使得这些架构难以部署在资源受限的设备上。此外，当前主流的轻量级骨干网络多为自然图像设计，在应对多尺度目标方面表现不佳，从而限制了其在遥感视觉任务中的表现。

本文提出了一种专门面向遥感视觉任务设计的轻量级骨干网络——LWGANet，其中引入了一种新颖的注意力机制模块：轻量级分组注意力（LWGA）模块。
LWGA 模块针对遥感图像特点，充分利用特征图中的冗余信息，在不增加额外复杂度和计算负担的前提下，从局部到全局提取多尺度空间信息，从而在保持高效率的同时实现精确的特征提取。

LWGANet 在场景分类、定向目标检测、语义分割和变化检测四类关键遥感视觉任务中共计12个数据集上进行了系统评估。
实验结果表明，LWGANet 在多个数据集上实现了先进性能，并且在高性能与低复杂度之间实现了优异的平衡，展现出良好的通用性与实用性。
总体而言，LWGANet 为资源受限场景下对遥感图像进行高效处理提供了一种新颖而有效的解决方案。


"""


class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            build_norm_layer(norm_layer, dim * 4)[1],
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x


class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), build_norm_layer(norm_layer, dim)[1], act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 使用简单的下采样替代antialiased_cnns.BlurPool
        self.max_m2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        )

        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = nn.BatchNorm2d(channel)

    def h_transform(self, x):
        B, C, H, W = x.shape

        # 创建输出张量
        output = torch.zeros((B, C, H, 2 * W - 1), device=x.device, dtype=x.dtype)

        # 填充操作
        for w in range(W):
            # 计算目标位置
            target_w = w
            # 复制特征
            output[..., target_w] = x[..., w]

        return output

    def inv_h_transform(self, x):
        B, C, H, W = x.shape
        # 原始宽度
        original_W = (W + 1) // 2

        # 创建输出张量
        output = torch.zeros((B, C, H, original_W), device=x.device, dtype=x.dtype)

        # 提取特征
        for w in range(original_W):
            target_w = 2 * w
            if target_w < W:
                output[..., w] = x[..., target_w]

        return output

    def v_transform(self, x):
        B, C, H, W = x.shape

        # 转置以便在高度维度上操作
        x = x.permute(0, 1, 3, 2)  # [B, C, W, H]

        # 创建输出张量
        output = torch.zeros((B, C, W, 2 * H - 1), device=x.device, dtype=x.dtype)

        # 填充操作
        for h in range(H):
            target_h = h
            output[..., target_h] = x[..., h]

        return output.permute(0, 1, 3, 2)  # 转置回来

    def inv_v_transform(self, x):
        B, C, H, W = x.shape
        # 原始高度
        original_H = (H + 1) // 2

        # 转置
        x = x.permute(0, 1, 3, 2)  # [B, C, W, H]

        # 创建输出张量
        output = torch.zeros((B, C, W, original_H), device=x.device, dtype=x.dtype)

        # 提取特征
        for h in range(original_H):
            target_h = 2 * h
            if target_h < H:
                output[..., h] = x[..., target_h]

        return output.permute(0, 1, 3, 2)  # 转置回来

    def forward(self, x):
        _B, _C, H, W = x.shape

        # 保存原始尺寸
        original_H, original_W = H, W

        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)

        # 获取下采样后的尺寸
        _, _, _H_down, _W_down = x_tem.shape

        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)

        # 使用安全的变换
        try:
            x_tem_h = self.h_transform(x_tem)
            x_h2_temp = self.H_att2(x_tem_h)
            x_h2 = self.inv_h_transform(x_h2_temp)

            x_tem_v = self.v_transform(x_tem)
            x_w2_temp = self.V_att2(x_tem_v)
            x_w2 = self.inv_v_transform(x_w2_temp)
        except Exception as e:
            # 如果变换失败，使用简化版本
            print(f"变换失败: {e}，使用简化版本")
            x_h2 = torch.zeros_like(x_h1)
            x_w2 = torch.zeros_like(x_w1)

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        # 调整注意力图尺寸
        if att.shape[-2:] != (original_H, original_W):
            att = F.interpolate(att, size=(original_H, original_W), mode="bilinear", align_corners=False)

        out = x * self.gate_fn(att)
        return out


class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


class D_GA(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x


class GA(nn.Module):
    def __init__(
        self, dim, head_dim=4, num_heads=None, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, proj_bias=False, **kwargs
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, _C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LWGA(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        stage=1,
        att_kernel=7,
        mlp_ratio=4.0,
        drop_path=0.1,
        act_layer=None,
        norm_layer=None,
        *args,
        **kwargs,
    ):
        """LWGA模块 - 适配YOLO框架 参数: c1: 输入通道数 (由YOLO自动传递) c2: 输出通道数，如果为None则使用c1 stage: 阶段，默认1 att_kernel: 注意力核大小，默认7
        mlp_ratio: MLP扩展比例，默认4.0 drop_path: DropPath比率，默认0.1 act_layer: 激活函数，默认nn.GELU
        norm_layer: 归一化层，默认BatchNorm2d.
        """
        super().__init__()

        # 处理参数
        if c2 is None:
            c2 = c1

        # 设置默认激活函数
        if act_layer is None:
            act_layer = nn.GELU
        elif isinstance(act_layer, str):
            # 从字符串转换为类
            if act_layer == "GELU":
                act_layer = nn.GELU
            elif act_layer == "ReLU":
                act_layer = nn.ReLU
            elif act_layer == "SiLU":
                act_layer = nn.SiLU
            else:
                act_layer = nn.GELU

        # 设置默认归一化层
        if norm_layer is None:
            norm_layer = dict(type="BN", requires_grad=True)
        elif isinstance(norm_layer, str):
            if norm_layer == "BN":
                norm_layer = dict(type="BN", requires_grad=True)
            elif norm_layer == "GN":
                norm_layer = dict(type="GN", num_groups=32)
            elif norm_layer == "LN":
                norm_layer = dict(type="LN")
            else:
                norm_layer = dict(type="BN", requires_grad=True)

        print(
            f"LWGA配置: 输入={c1}, 输出={c2}, stage={stage}, att_kernel={att_kernel}, "
            f"mlp_ratio={mlp_ratio}, drop_path={drop_path}"
        )

        self.stage = stage
        self.dim_split = c2 // 4
        self.drop_path_rate = drop_path

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP
        mlp_hidden_dim = int(c2 * mlp_ratio)

        mlp_layer = [
            nn.Conv2d(c2, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, c2, 1, bias=False),
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        # 各注意力模块
        self.PA = PA(self.dim_split, norm_layer, act_layer)
        self.LA = LA(self.dim_split, norm_layer, act_layer)
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)

        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)
            self.norm = build_norm_layer(norm_layer, self.dim_split)[1]
        else:
            self.GA12 = GA12(self.dim_split, act_layer)
            self.norm = build_norm_layer(norm_layer, self.dim_split)[1]

        self.norm1 = build_norm_layer(norm_layer, c2)[1]
        self.drop_path = DropPath(drop_path)

        # 如果需要通道数适配
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1)
        else:
            self.channel_adapter = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # 适配通道数
        x = self.channel_adapter(x)

        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)

        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)

        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))

        x_att = torch.cat((x1, x2, x3, x4), 1)
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x


if __name__ == "__main__":
    # 定义输入特征图大小：Batch=1, Channel=64, H=32, W=32
    input = torch.randn(1, 64, 32, 32)
    # 初始化 LWGA
    block = LWGA(
        dim=64,  # 输入特征图的通道数
        stage=1,  # 网络的stage，决定使用哪种GA（GA12）
        att_kernel=7,  # MRA中的注意力卷积核大小
        mlp_ratio=4.0,  # MLP通道扩展比例
        drop_path=0.1,  # DropPath
        act_layer=nn.GELU,  # 激活函数
        norm_layer=dict(type="BN", requires_grad=True),  # 使用 BatchNorm
    )
    # 前向传播
    output = block(input)
    # 输出张量信息
    print(f"Ai缝合怪永久更新中—LWGA Input shape:  {input.shape}")
    print(f"Ai缝合怪永久更新中—LWGA Output shape: {output.shape}")
