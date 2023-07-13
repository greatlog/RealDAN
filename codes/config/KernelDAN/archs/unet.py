import torch
import torch.nn as nn
import torchvision

from utils.registry import ARCH_REGISTRY


class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

@ARCH_REGISTRY.register()
class Unet(nn.Module):
    def __init__(self, net_type, channels=[64, 128, 256, 512]):
        super().__init__()

        base_model = getattr(torchvision.models, net_type)(pretrained=False)
        self.base_layers = list(base_model.children())[:-2]

        self.layer1 = nn.Sequential(*self.base_layers[:3])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])

        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]

        self.decode4 = Decoder(channels[-1], channels[-2] * 2, channels[-2])
        self.decode3 = Decoder(channels[-2], channels[-2] + channels[-3], channels[-2])
        self.decode2 = Decoder(channels[-2], channels[-3] + channels[-4], channels[-3])
        self.decode1 = Decoder(channels[-3], channels[-4] * 2, channels[-4])

        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            )

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        return d0