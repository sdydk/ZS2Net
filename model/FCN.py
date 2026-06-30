import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.vgg import VGGNet16
from torchvision.models import vgg16
class FCN32s(nn.Module):
    def __init__(self, n_class, pretrained_net):
        super(FCN32s, self).__init__()
        self.n_class = n_class
        self.features = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        x4 = features[4]

        x = self.bn1(self.relu(self.deconv1(x4)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))
        x = self.classifier(x)
        return x


class FCN16s(nn.Module):
    def __init__(self, n_class, pretrained_net):
        super(FCN16s, self).__init__()
        self.n_class = n_class
        self.features = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        x5 = features[4]
        x4 = features[3]

        x = self.relu(self.deconv1(x5))
        x = self.bn1(x + x4)
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))
        x = self.classifier(x)
        return x

class FCN8s(nn.Module):
    def __init__(self, n_class, pretrained_net):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.features = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        x5 = features[4]
        x4 = features[3]
        x3 = features[2]

        x = self.relu(self.deconv1(x5))
        x = self.bn1(x + x4)

        x = self.relu(self.deconv2(x))
        x = self.bn2(x + x3)

        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))
        x = self.classifier(x)
        return x


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

class FCNs(nn.Module):
    def __init__(self, args):
        super(FCNs, self).__init__()
        backbone = 'vgg16'
        self.num_classes = args.num_class
        self.pretrained = args.pretrained
        self.ranges = ranges[backbone]

        if backbone == "vgg16":
            if self.pretrained:
                self.features = vgg16(pretrained=True)
                del self.features.classifier
                del self.features.avgpool
            else:
                self.features = VGGNet16()

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.classifier = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, x):
        if self.pretrained:
            features = {}
            for idx, (begin, end) in enumerate(self.ranges):
                for layer in range(begin, end):
                    x = self.features.features[layer](x)
                features["x%d"%(idx+1)] = x
        else:
            features = self.features(x)

        x = self.bn1(self.relu1(self.deconv1(features['x5'])) + features['x4'])
        x = self.bn2(self.relu2(self.deconv2(x)) + features['x3'])
        x = self.bn3(self.relu3(self.deconv3(x)) + features['x2'])
        x = self.bn4(self.relu4(self.deconv4(x)) + features['x1'])
        x = self.bn5(self.relu5(self.deconv5(x)))

        x = self.classifier(x)
        return x
class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = F.softmax(dist, dim=-1)
        attn = torch.bmm(dist, v)
        return attn
