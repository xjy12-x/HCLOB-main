import torch
from torch import nn

from ultralytics.nn.modules import C3


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _N, C, _H, _W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        _N, C, _H, _W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


############GC#####################
class DWConv(nn.Module):  # 定义一个深度可分离卷积类
    def __init__(self, dim=768):  # 初始化函数，dim是输入通道数，默认值为768
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )  # 定义一个深度可分离卷积层

    def forward(self, x, H, W):  # 前向传播函数，x是输入，H和W是目标的高度和宽度
        B, _N, C = x.shape  # 获取输入的batch大小B，特征数N，通道数C
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()  # 将输入的维度重排为适合卷积操作的4D张量
        x = self.dwconv(x)  # 进行深度可分离卷积
        x = x.flatten(2).transpose(1, 2)  # 将卷积结果展平，恢复回原来的维度
        return x


class GatedConvolutionalLinearModule(nn.Module):  # 定义一个门控卷积线性模块类
    def __init__(self, in_channels, drop=0.1):  # 初始化函数，in_channels是输入通道数，drop是Dropout的比例
        super().__init__()

        hidden_channels = int(2 * in_channels)  # 定义隐藏通道数为2倍的输入通道数
        self.fc1 = nn.Linear(in_channels, hidden_channels * 2)  # 定义一个全连接层fc1，将输入映射到隐藏空间
        self.dwconv = DWConv(hidden_channels)  # 定义一个深度可分离卷积层，输入通道数为hidden_channels
        self.act = nn.GELU()  # 定义一个GELU激活函数
        self.fc2 = nn.Linear(hidden_channels, in_channels)  # 定义一个全连接层fc2，将隐藏层映射回输入通道数
        self.drop = nn.Dropout(drop)  # 定义一个Dropout层，用于防止过拟合

    def forward(self, x):  # 前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # 将输入张量重排为3D张量，形状为(B, H*W, C)
        x1, x2 = self.fc1(x).chunk(2, dim=-1)  # 将输入传入fc1层，并分成两个部分x1和x2
        x = self.act(self.dwconv(x1, h, w)) * self.dwconv(x2, h, w)  # 经过深度可分离卷积并计算门控输出
        x = self.drop(x)  # 使用Dropout
        x = self.fc2(x)  # 经过第二个全连接层
        x = self.drop(x)  # 使用Dropout
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # 将输出张量重排回4D张量
        return x


###########################################################################################
class Branch(nn.Module):
    """Branch that lasts lonly the dilated convolutions."""

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dw_channel,
                out_channels=self.dw_channel,
                kernel_size=3,
                padding=dilation,
                stride=1,
                groups=self.dw_channel,
                bias=True,
                dilation=dilation,
            )  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class GCDB(nn.Module):
    """Change this block using Branch."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=None, extra_depth_wise=False):
        if dilations is None:
            dilations = [1]
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=self.dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )
        self.extra_conv = (
            nn.Conv2d(
                self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1
            )
            if extra_depth_wise
            else nn.Identity()
        )  # optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=self.dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )
        self.sg1 = GatedConvolutionalLinearModule(self.dw_channel)
        self.sg2 = GatedConvolutionalLinearModule(FFN_Expand * c)
        self.conv3 = nn.Conv2d(
            in_channels=self.dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    #        self.adapter = Adapter(c, ffn_channel=None)

    #        self.use_adapters = False

    #    def set_use_adapters(self, use_adapters):
    #        self.use_adapters = use_adapters

    def forward(self, inp, adapter=None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]
        x = y + x * self.gamma


#################C3k2_GCDB######################################


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_GCDB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = GCDB(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C2f_GCDB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_GCDB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_GCDB(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_GCDB(C2f_GCDB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_GCDB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
