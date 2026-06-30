import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.ResNet50 import ResNet50
from torchvision.models import resnet50
class FPNs(nn.Module):
    def __init__(self, num_class, pretrained = True):
        super(FPNs, self).__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.encoder = resnet50(pretrained=self.pretrained)
            del self.encoder.avgpool
            del self.encoder.fc
        else:
            self.encoder = ResNet50()
        self.num_class = num_class

        self.middle5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1)
        self.middle4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
        self.middle3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.middle2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)

        # remove aliasing effect of up_sample
        self.final2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.final3 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.final4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.final5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=1, stride=2)

        # group normalization (num_groups, num_channels)  保持通道之间的相对关系
        # 将输入数据分成多个组，并分别计算每个组的均值和方差，使用这些均值和方差对每组的元素进行归一化
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)
        self.segmentation = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=self.num_class, kernel_size=1, stride=1)

    def forward(self, x):
        if self.pretrained:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            c2 = self.encoder.layer1(x)
            c3 = self.encoder.layer2(c2)
            c4 = self.encoder.layer3(c3)
            c5 = self.encoder.layer4(c4)
        else:
            out = self.encoder(x)
            c2, c3, c4, c5 = out['x1'], out['x2'], out['x3'], out['x4']

        c2 = self.middle2(c2)
        c3 = self.middle3(c3)
        c4 = self.middle4(c4)
        c5 = self.middle5(c5)

        c5_up = F.interpolate(c5, size=(c4.size(2), c4.size(3)), mode='bilinear', align_corners=True)
        c4 = torch.cat([c4, c5_up], dim=1)

        c4_up = F.interpolate(c4, size=(c3.size(2), c3.size(3)), mode='bilinear', align_corners=True)
        c3 = torch.cat([c3, c4_up], dim=1)

        c3_up = F.interpolate(c3, size=(c2.size(2), c2.size(3)), mode='bilinear', align_corners=True)
        c2 = torch.cat([c2, c3_up], dim=1)

        p2 = self.final2(c2)
        p3 = self.final3(c3)
        p4 = self.final4(c4)
        p5 = self.final5(c5)
        p6 = self.max_pool1(p5)

        # semantic segmentation
        s5 = F.interpolate(
            F.relu(
                self.gn2(self.conv2(p5))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )
        s5 = F.interpolate(
            F.relu(
                self.gn2(self.conv2(s5))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )
        s5 = F.interpolate(
            F.relu(
                self.gn1(self.segmentation(s5))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )
        s4 = F.interpolate(
            F.relu(
                self.gn2(self.conv2(p4))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )
        s4 = F.interpolate(
            F.relu(
                self.gn1(self.segmentation(s4))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )

        s3 = F.interpolate(
            F.relu(
                self.gn1(self.segmentation(p3))
            ),
            size=(p2.size(2), p2.size(3)),
            mode='bilinear',
            align_corners=True
        )
        s2 = F.relu(self.gn1(self.segmentation(p2)))

        result = F.interpolate(
            self.conv3(s2 + s3 + s4 + s5), size=(p2.size(2)*4, p2.size(3)*4), mode='bilinear', align_corners=True
        )
        return result