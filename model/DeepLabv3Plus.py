import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101
from model.backbone.ResNet101 import ResNet101
class DeepLabv3plus(nn.Module):
    def __init__(self, n_class, in_channel, pretrained=True):
        super(DeepLabv3plus, self).__init__()
        self.n_class = n_class
        self.in_channel = in_channel
        self.pretrained = pretrained
        self.encoder = Encoder(self.pretrained)
        self.decoder = Decoder(in_channel=256, out_channel=256, n_class=self.n_class)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x_aspp, xl = self.encoder(x)
        x = self.decoder(x_aspp, xl)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        self.pretrained = pretrained
        self.dcnn = DCNN(model_name='ResNet', pretrained=self.pretrained)
        self.aspp = ASPP(in_channel=2048, out_channel=256)
    def forward(self, x):
        x, xl = self.dcnn(x)
        x_aspp = self.aspp(x)
        return x_aspp, xl


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, n_class):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=48, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=48+256, out_channels=out_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=n_class, kernel_size=1, stride=1)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
    def forward(self, x_aspp, xl):
        x1 = self.conv1(xl)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = F.interpolate(x_aspp, size=(xl.size(2), xl.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.conv3(x)

        # x = self.upsample(x)
        return x
class DCNN(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(DCNN, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        if self.model_name == 'ResNet':
            if self.pretrained:
                self.feature = resnet101(pretrained=True)
                # for param in self.feature.parameters():
                #     param.requires_grad = False
                del self.feature.avgpool
                del self.feature.fc
            else:
                self.feature = ResNet101()
        elif self.model_name == 'Xception':
            self.feature = Xception()
        else:
            print('Model is not exit!!!')

    def forward(self, x):
        if self.pretrained:
            x = self.feature.conv1(x)
            x = self.feature.bn1(x)
            x = self.feature.relu(x)
            x = self.feature.maxpool(x)
            x1 = self.feature.layer1(x)

            x = self.feature.layer2(x1)
            x = self.feature.layer3(x)
            x = self.feature.layer4(x)
        else:
            x, x1 = self.feature(x)

        return x, x1

class ASPP(nn.Module):
    def __init__(self, in_channel=3, out_channel=256):
        super(ASPP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=6, stride=1, dilation=6),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=12, stride=1, dilation=12),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=18, stride=1, dilation=18),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_channels=out_channel*5, out_channels=out_channel, kernel_size=1, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x1 = self.layer1(x) #[1, 2048, 14, 14] -> [1, 256, 8, 8]
        x2 = self.layer2(x) #[1, 2048, 14, 14] -> [1, 256, 2, 2]
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = F.interpolate(self.layer5(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# class ResNet101(nn.Module):
#     def __init__(self):
#         super(ResNet101, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = nn.Sequential(
#             Bottlenck(in_channel=64, out_channel=256, if_sample=True),
#             Bottlenck(in_channel=256, out_channel=256, if_sample=False),
#             Bottlenck(in_channel=256, out_channel=256, if_sample=False)
#         )
#
#         self.layer2 = nn.Sequential(
#             Bottlenck(in_channel=256, out_channel=512, if_sample=True, stride=2),
#             Bottlenck(in_channel=512, out_channel=512, if_sample=False),
#             Bottlenck(in_channel=512, out_channel=512, if_sample=False),
#             Bottlenck(in_channel=512, out_channel=512, if_sample=False)
#         )
#
#         self.layer3 = nn.Sequential(
#             Bottlenck(in_channel=512, out_channel=1024, if_sample=True, stride=2),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
#             Bottlenck(in_channel=1024, out_channel=1024, if_sample=False)
#         )
#         self.layer4 = nn.Sequential(
#             Bottlenck(in_channel=1024, out_channel=2048, if_sample=False),
#             Bottlenck(in_channel=2048, out_channel=2048, if_sample=False),
#             Bottlenck(in_channel=2048, out_channel=2048, if_sample=False)
#         )
#     def forward(self, x):
#         x = self.conv1(x) #[1, 64, 112, 112] -> [1, 64, 112, 112]
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x) #[1, 64, 112, 112] -> [1, 64, 56, 56]
#         x1 = self.layer1(x)
#
#         x = self.layer2(x1)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x, x1
# class Bottlenck(nn.Module):
#     def __init__(self, in_channel, out_channel, if_sample=True, stride=1):
#         super(Bottlenck, self).__init__()
#         self.if_sample = if_sample,
#         self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride)
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel//4, kernel_size=1, stride=1)
#         self.bn1 = nn.BatchNorm2d(out_channel//4)
#         self.conv2 = nn.Conv2d(in_channels=out_channel//4, out_channels=out_channel//4, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel//4)
#         self.conv3 = nn.Conv2d(in_channels=out_channel//4, out_channels=out_channel, kernel_size=1, stride=1)
#         self.bn3 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#         if self.if_sample[0]==True:
#             self.sample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channel)
#             )
#
#     def forward(self, x):
#         # [1, 64, 56, 56] -> [1, 256, 56, 56]
#         identity = x
#         x = self.conv1(x) #[1, 64, 56, 56] -> [1, 64, 56, 56]
#         x = self.bn1(x)
#         x = self.conv2(x) #[1, 64, 56, 56]
#         x = self.bn2(x)
#         x = self.conv3(x) #[1, 256, 56, 56]
#         x = self.bn3(x)
#
#         if self.if_sample[0]==True:
#             identity = self.sample(identity)
#         if self.if_sample[0] == False:
#             identity = self.conv(identity)
#         x += identity
#         x = self.relu(x)
#         return x
class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
    def forward(self, x):

        return x
