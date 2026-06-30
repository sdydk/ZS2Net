import torch
import torch.nn as nn
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv_relu(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_class=19):
        super(UNet, self).__init__()
        self.n_class = n_class
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.up(x5)

        x5 = torch.cat([x4, x5], dim=1)
        x5 = self.up1(x5)

        x5 = torch.cat([x3, x5], dim=1)
        x5 = self.up2(x5)

        x5 = torch.cat([x2, x5], dim=1)
        x5 = self.up3(x5)

        x5 = torch.cat([x1, x5], dim=1)
        x5 = self.conv_2(x5, is_pool=False)
        x5 = self.last(x5)

        return x5
