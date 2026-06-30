import torch
from torch import nn
import torch.nn.functional as F
from model.backbone.HRNets import HighResolutionNet
class OCRNets(nn.Module):
    def __init__(self, args):
        super(OCRNets, self).__init__()
        self.num_classes = args.num_class
        self.pretrained = args.pretrained
        self.scale = 1
        self.ocr_mid_channels = args.ocr_mid_channels
        self.ocr_key_channels = args.ocr_key_channels
        self.backbone = HighResolutionNet()
        if self.pretrained:
            weights = args.weights
            self.backbone.load_state_dict(torch.load(weights), strict=False)
        self.pixel_representation = nn.Sequential(
            nn.Conv2d(in_channels=720, out_channels=self.ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ocr_mid_channels),
            nn.ReLU(inplace=True)
        )
        self.soft_object_regions = nn.Sequential(
            nn.Conv2d(in_channels=720, out_channels=720, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(720),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=720, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.ocr_mid_channels, out_channels=self.ocr_key_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.ocr_key_channels, out_channels=self.ocr_key_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_key_channels),
            nn.ReLU(),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.ocr_mid_channels, out_channels=self.ocr_key_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.ocr_key_channels, out_channels=self.ocr_key_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_key_channels),
            nn.ReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.ocr_mid_channels, out_channels=self.ocr_key_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_key_channels),
            nn.ReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.ocr_key_channels, out_channels=self.ocr_mid_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(self.ocr_mid_channels),
            nn.ReLU(),
        )
        self.conv_bn_dropout=nn.Sequential(
            nn.Conv2d(in_channels=2*self.ocr_mid_channels, out_channels=self.ocr_mid_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ocr_mid_channels),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.segmentation_head = nn.Conv2d(in_channels=self.ocr_mid_channels, out_channels=self.num_classes,
                                           kernel_size=1, stride=1, padding=0, bias=True)
    def _get_sim_map(self, feats, proxy):
        x = feats
        b, h, w = x.size(0), x.size(2), x.size(3)
        query = self.f_pixel(x).view(b, self.ocr_key_channels, -1).permute(0, 2, 1)
        key = self.f_object(proxy).view(b, self.ocr_key_channels, -1)
        value = self.f_down(proxy).view(b, self.ocr_key_channels, -1).permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.ocr_key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map, value, b

    def _get_context(self, sim_map, feats, value, batch_size):
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.ocr_key_channels, *feats.size()[2:])
        context = self.f_up(context)
        output = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return output
    def _object_region_representations(self, feats, probs):
        b, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(b, c, -1)
        feats = feats.view(b, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        proxy = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        return proxy

    def forward(self, x):
        H, W = x.shape[-2:]
        features = self.backbone(x)
        probs = self.soft_object_regions(features)
        feats = self.pixel_representation(features)
        proxy = self._object_region_representations(feats=feats, probs=probs)
        sim_map, value, b = self._get_sim_map(feats, proxy)
        result = self._get_context(sim_map, feats, value, b)
        result = self.segmentation_head(result)

        result = F.interpolate(result, size=(H, W), mode='bilinear', align_corners=True)
        probs = F.interpolate(probs, size=(H, W), mode='bilinear', align_corners=True)
        return result, probs