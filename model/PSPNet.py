import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from model.backbone.ResNet101 import ResNet101
from model.backbone.MobileNetV2 import MobileNetV2
class PSPNets(nn.Module):
    def __init__(self, in_channel, pretrained, name, out_channels):
        super(PSPNets, self).__init__()
        self.pretrained = pretrained
        self.name = name
        self.out_channels = out_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        if self.name == "ResNet101":
            if self.pretrained:
                self.backbone = models.resnet101(pretrained=True)
                del self.backbone.avgpool
                del self.backbone.fc
            else:
                self.backbone = ResNet101()
        else:
            if self.pretrained:
                self.backbone = models.mobilenet_v2(pretrained=True)
            else:
                self.backbone = MobileNetV2(in_channel=32)

        self.ppm = PyramidPoolingModule()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        if self.name == "ResNet101":
            if self.pretrained:
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x_aux = self.backbone.layer1(x)
                x = self.backbone.layer2(x_aux)
                x = self.backbone.layer3(x)
                x1 = self.backbone.layer4(x)
            else:
                x, x1 = self.backbone(x)
        else:
            if self.pretrained:
                x1, x_aux = self.backbone(x)
            else:
                x = self.layer1(x)
                x1, x_aux = self.backbone(x)
        x = self.ppm(x1)
        x = self.layer2(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
class PyramidPoolingModule(nn.Module):
    def __init__(self):
        super(PyramidPoolingModule, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=80, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=2368, out_channels=80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        identity = x
        x1 = self.pool1(x)
        x1 = self.conv_layer(x1)
        x1 = F.interpolate(x1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x2 = self.pool2(x)
        x2 = self.conv_layer(x2)
        x2 = F.interpolate(x2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x3 = self.pool3(x)
        x3 = self.conv_layer(x3)
        x3 = F.interpolate(x3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x4 = self.pool4(x)
        x4 = self.conv_layer(x4)
        x4 = F.interpolate(x4, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([identity, x1, x2, x3, x4], dim=1)

        x = self.conv_layer2(x)
        return x