import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.ResNet50 import ResNet50
from torchvision.models import resnet50, resnet101
from thop import profile
from thop import clever_format
from einops import rearrange
import pywt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                             kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                             padding=1, stride=1, groups=in_channels * 2, bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')

        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)

        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)

        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)

        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)

        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class tDFFT(nn.Module):
    def __init__(self, dim, groups):
        super(tDFFT, self, ).__init__()
        self.dim = dim
        self.groups = groups
        self.gn = nn.GroupNorm(self.dim, self.dim)
        self.fourier = FourierUnit(in_channels=self.dim, out_channels=self.dim, groups=self.groups)
        self.fourier_fina = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1),
            nn.GELU()
        )
        self.conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fourier_fina(self.fourier(x))
        return x


class CAM(nn.Module):
    def __init__(self, dim):
        super(CAM, self).__init__()
        self.dim = dim
        self.chat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * 2, 2 * self.dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.dim // 2, self.dim * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.chat(x) * x
        return x
class tDWT(nn.Module):
    def __init__(self, dim):
        super(tDWT, self).__init__()
        self.dim = dim
        self.ca = CAM(dim=64)
        self.conv = nn.Conv2d(in_channels=self.dim, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        coeffs_haar = pywt.dwt2(x.detach().cpu().numpy(), 'haar')
        x_cA, (x_cH, x_cV, x_cD) = coeffs_haar
        x_cA = self.conv(torch.tensor(x_cA, device=self.conv.weight.device))
        x_cH = self.conv(torch.tensor(x_cH, device=self.conv.weight.device))
        x_cV = self.conv(torch.tensor(x_cV, device=self.conv.weight.device))
        x_cD = self.conv(torch.tensor(x_cD, device=self.conv.weight.device))
        x = torch.cat([x_cA, x_cH, x_cV, x_cD], dim=1)
        x = self.ca(x)
        return x

class EEM(nn.Module):
    def __init__(self, dim):
        super(EEM, self).__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(3, 3),
                               padding=1, stride=1, groups=self.dim)

        self.conv1_1 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 5),
                                 padding=(0, 2), stride=1, groups=self.dim)
        self.conv1_2 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(5, 1),
                                 padding=(2, 0), stride=1, groups=self.dim)

        self.conv2_1 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 7),
                                 padding=(0, 3), stride=1, groups=self.dim)
        self.conv2_2 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(7, 1),
                                 padding=(3, 0), stride=1, groups=self.dim)

        self.conv3_1 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 9),
                                 padding=(0, 4), stride=1, groups=self.dim)
        self.conv3_2 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(9, 1),
                                 padding=(4, 0), stride=1, groups=self.dim)

        self.conv_fina = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = x
        x = self.conv0(x)

        x_1_1 = self.conv1_1(x)
        x_1_2 = self.conv1_2(x_1_1)

        x_2_1 = self.conv2_1(x)
        x_2_2 = self.conv2_2(x_2_1)

        x_3_1 = self.conv3_1(x)
        x_3_2 = self.conv3_2(x_3_1)

        x = x + x_1_2 + x_2_2 + x_3_2
        x = self.conv_fina(x)
        x = x1 + x
        return x

class ZS2Net(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(ZS2Net, self).__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.encoder = resnet50(pretrained=self.pretrained)
            del self.encoder.avgpool
            del self.encoder.fc
        else:
            self.encoder = ResNet50()
        self.num_class = num_class
        self.eem = EEM(dim=self.num_class)
        self.tdwt = tDWT(dim=3)
        self.tdfft0 = tDFFT(dim=2048, groups=4)

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
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)
        self.segmentation = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=self.num_class, kernel_size=1, stride=1)

    def forward(self, x):
        x_wt = self.tdwt(x)

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

        copy_c5 = self.tdfft0(c5)

        c2 = self.middle2(c2)
        c3 = self.middle3(c3)
        c4 = self.middle4(c4)
        c5 = self.middle5(copy_c5)

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
            size=(x_wt.size(2), x_wt.size(3)),
            mode='bilinear',
            align_corners=True)

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
            size=(x_wt.size(2), x_wt.size(3)),
            mode='bilinear',
            align_corners=True
        )

        s3 = F.interpolate(
            F.relu(
                self.gn1(self.segmentation(p3))
            ),
            size=(x_wt.size(2), x_wt.size(3)),
            mode='bilinear',
            align_corners=True
        )

        s2 = F.relu(self.gn1(self.segmentation(p2)))
        s2 = F.interpolate(s2, size=(x_wt.size(2), x_wt.size(3)), mode='bilinear', align_corners=True)
        x = self.conv3(x_wt + s2 + s3 + s4 + s5)
        x_eem = self.eem(x)

        result = F.interpolate(x_eem, size=(p2.size(2) * 4, p2.size(3) * 4), mode='bilinear', align_corners=True)
        return result