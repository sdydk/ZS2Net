import torch
from torch import nn
class MobileNetV2(nn.Module):
    def __init__(self, in_channel, ):
        super(MobileNetV2, self).__init__()
        self.dw = _depthwise_conv_block(in_channel=32, out_channel=32, kernel_size=3, stride=1, pad=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU6(inplace=True)

        self.layer1 = DepthWiseBottlenck(in_channel=16, out_channel1=96, out_channel=24, stride=2)
        self.layer2 = DepthWiseBottlenck(in_channel=24, out_channel1=144, out_channel=24, stride=1)

        self.layer3 = DepthWiseBottlenck(in_channel=24*2, out_channel1=144, out_channel=32, stride=2)
        self.layer4 = DepthWiseBottlenck(in_channel=32, out_channel1=192, out_channel=32, stride=1)
        self.layer5 = DepthWiseBottlenck(in_channel=32*2, out_channel1=192, out_channel=32, stride=1)

        self.layer6 = DepthWiseBottlenck(in_channel=32*3, out_channel1=192, out_channel=64, stride=2)
        self.layer7 = DepthWiseBottlenck(in_channel=64, out_channel1=384, out_channel=64, stride=1)
        self.layer8 = DepthWiseBottlenck(in_channel=64*2, out_channel1=384, out_channel=64, stride=1)
        self.layer9 = DepthWiseBottlenck(in_channel=64*3, out_channel1=384, out_channel=64, stride=1)

        self.layer10 = DepthWiseBottlenck(in_channel=64*4, out_channel1=384, out_channel=96, stride=1)
        self.layer11 = DepthWiseBottlenck(in_channel=96, out_channel1=576, out_channel=96, stride=1)
        self.layer12 = DepthWiseBottlenck(in_channel=96*2, out_channel1=576, out_channel=96, stride=1)

        self.layer13 = DepthWiseBottlenck(in_channel=96*3, out_channel1=576, out_channel=160, stride=1)
        self.layer14 = DepthWiseBottlenck(in_channel=160, out_channel1=960, out_channel=160, stride=1)
        self.layer15 = DepthWiseBottlenck(in_channel=160*2, out_channel1=960, out_channel=160, stride=1)

        self.layer16 = DepthWiseBottlenck(in_channel=160*3, out_channel1=960, out_channel=320, stride=1)

    def forward(self, x):
        x = self.dw(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = torch.cat([x1, x], dim=1)

        x2 = self.layer3(x)
        x3 = self.layer4(x2)

        x4 = torch.cat([x2, x3], dim=1)
        x = self.layer5(x4)
        x = torch.cat([x4, x], dim=1)

        x5 = self.layer6(x)
        x6 = self.layer7(x5)
        x7 = torch.cat([x5, x6], dim=1)

        x8 = self.layer8(x7)
        x9 = torch.cat([x7, x8], dim=1)
        x10 = self.layer9(x9)
        x = torch.cat([x9, x10], dim=1)

        x11 = self.layer10(x)
        x12 = self.layer11(x11)
        x13 = torch.cat([x11, x12], dim=1)
        x14 = self.layer12(x13)
        x_aux = torch.cat([x13, x14], dim=1)

        x15 = self.layer13(x_aux)
        x16 = self.layer14(x15)
        x17 = torch.cat([x15, x16], dim=1)
        x18 = self.layer15(x17)

        x = torch.cat([x17, x18], dim=1)
        x = self.layer16(x)
        return x, x_aux

class DepthWiseBottlenck(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel, stride):
        super(DepthWiseBottlenck, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU6(inplace=True)
        )
        self.layer2 = _depthwise_conv_block(
            in_channel=out_channel1,
            out_channel=out_channel1,
            stride=stride,
            kernel_size=3,
            pad=0
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel1, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class _depthwise_conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, pad=1):
        super(_depthwise_conv_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channel
        )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x