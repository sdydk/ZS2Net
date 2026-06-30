from torch import nn
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            Bottlenck(in_channel=64, out_channel=256, if_sample=True),
            Bottlenck(in_channel=256, out_channel=256, if_sample=False),
            Bottlenck(in_channel=256, out_channel=256, if_sample=False)
        )

        self.layer2 = nn.Sequential(
            Bottlenck(in_channel=256, out_channel=512, if_sample=True, stride=2),
            Bottlenck(in_channel=512, out_channel=512, if_sample=False),
            Bottlenck(in_channel=512, out_channel=512, if_sample=False),
            Bottlenck(in_channel=512, out_channel=512, if_sample=False)
        )

        self.layer3 = nn.Sequential(
            Bottlenck(in_channel=512, out_channel=1024, if_sample=True, stride=2),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False),
            Bottlenck(in_channel=1024, out_channel=1024, if_sample=False)
        )
        self.layer4 = nn.Sequential(
            Bottlenck(in_channel=1024, out_channel=2048, if_sample=False),
            Bottlenck(in_channel=2048, out_channel=2048, if_sample=False),
            Bottlenck(in_channel=2048, out_channel=2048, if_sample=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)

        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, x1
class Bottlenck(nn.Module):
    def __init__(self, in_channel, out_channel, if_sample=True, stride=1):
        super(Bottlenck, self).__init__()
        self.if_sample = if_sample,
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel//4, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel//4)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel//4,
            out_channels=out_channel//4,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel//4)
        self.conv3 = nn.Conv2d(in_channels=out_channel//4, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        if self.if_sample[0]==True:
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.if_sample[0]==True:
            identity = self.sample(identity)
        if self.if_sample[0] == False:
            identity = self.conv(identity)
        x += identity
        x = self.relu(x)
        return x