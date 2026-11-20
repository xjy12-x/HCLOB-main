# ultralytics/nn/newmodules/converse2D.py
import torch
from torch import nn


class Converse2D(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=5, scale=2, padding=4, padding_mode="circular", eps=1e-5):
        super().__init__()

        if c2 is None or c2 == -1:
            c2 = c1

        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # 确保通道相等
        if self.out_channels != self.in_channels:
            print(f"警告: 调整输出通道 {c2} -> {c1}")
            c2 = c1
            self.out_channels = c1

        # 使用标准的Parameter，避免复杂的初始化
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))

        # 使用简单的初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """简化权重初始化."""
        with torch.no_grad():
            # 避免在初始化时创建复杂的计算图
            weight_flat = self.weight.data.view(1, self.in_channels, -1)
            # 使用简单的softmax，避免复杂的操作
            weight_softmax = torch.nn.functional.softmax(weight_flat, dim=-1)
            self.weight.data = weight_softmax.view(1, self.in_channels, self.kernel_size, self.kernel_size)

    def forward(self, x):
        # 保存输入的数据类型
        input_dtype = x.dtype

        # 确保在FFT计算中使用float32
        if input_dtype == torch.float16:
            x = x.to(torch.float32)

        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps

        if self.padding > 0:
            x = nn.functional.pad(
                x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0
            )

        _, _, h, w = x.shape
        STy_AiFengheguai = self.upsample(x, scale=self.scale)

        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

        FB = self.p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)

        # 确保FFT操作在float32上进行
        FBFy = FBC * torch.fft.fftn(STy_AiFengheguai, dim=(-2, -1))

        FR = FBFy + torch.fft.fftn(biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR_AiFengheguai = FBR.div(invW + biaseps)
        FCBinvWBR = FBC * invWBR_AiFengheguai.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[
                ...,
                self.padding * self.scale : -self.padding * self.scale,
                self.padding * self.scale : -self.padding * self.scale,
            ]

        # 转换回原始数据类型
        if input_dtype == torch.float16:
            out = out.to(torch.float16)

        return out

    def splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading_dims, scale, W_s, scale, H_s)
        permute_order = [
            *list(range(len(leading_dims))),
            len(leading_dims) + 1,
            len(leading_dims) + 3,
            len(leading_dims),
            len(leading_dims) + 2,
        ]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        # 确保在FFT计算中使用float32
        if psf.dtype == torch.float16:
            psf = psf.to(torch.float32)

        otf = torch.zeros(psf.shape[:-2] + shape, device=psf.device, dtype=torch.complex64)
        otf[..., : psf.shape[-2], : psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf

    def upsample(self, x, scale=3):
        st = 0
        # 保持数据类型一致性
        z = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale), device=x.device, dtype=x.dtype
        )
        z[..., st::scale, st::scale].copy_(x)
        return z
