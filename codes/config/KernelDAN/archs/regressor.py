import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from kornia.color import rgb_to_ycbcr

from utils.registry import ARCH_REGISTRY
from .unet import Unet


@ARCH_REGISTRY.register()
class KernelRegressor(nn.Module):
    def __init__(self, num_elements, net_type, crop_size=64, channels=[64, 128, 256, 512]):
        super().__init__()

        self.model = Unet(net_type, channels)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Conv2d(channels[0], num_elements * 2, 1, 1, 0, bias=False)

        self.crop = torchvision.transforms.CenterCrop(crop_size)
        
    def forward(self, x):

        x = self.crop(x)

        f = self.model(x)
        f = self.avg(f)
        f = self.tail(f).squeeze(-1).squeeze(-1)

        kernel, kernel_unc = f.chunk(2, dim=1)

        return kernel, kernel_unc

@ARCH_REGISTRY.register()
class NoiseRegressor(nn.Module):
    def __init__(self, net_type, crop_size, channels=[64, 128, 256, 512]):
        super().__init__()

        self.model = Unet(net_type, channels)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Conv2d(channels[0], 4, 1, 1, 0, bias=False)

        self.crop = torchvision.transforms.CenterCrop(crop_size)

    def forward(self, x):

        x = self.crop(x)

        f = self.model(x)
        f = self.avg(f)
        f = self.tail(f).squeeze(-1).squeeze(-1)

        sigma_p, unc_p, sigma_g, unc_g = f.chunk(4, dim=1)

        return sigma_p, unc_p, sigma_g, unc_g


@ARCH_REGISTRY.register()
class JPEGRegressor(nn.Module):
    def __init__(self, net_type):
        super().__init__()

        self.model = getattr(torchvision.models, net_type)(
            pretrained=False, num_classes=4)
        
    def forward(self, x):

        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0
        
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

        x = rgb_to_ycbcr(x)
        b, c, h, w = x.shape

        x = x.view(b, c, h//8, 8, w//8, 8).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b * h//8 * w // 8, c, 8, 8)

        is_jpeg, quality, quality_unc = self.model(x).view(b, h//8 * w//8, -1).mean(1).split([2, 1, 1], dim=1)

        return is_jpeg, quality, quality_unc
