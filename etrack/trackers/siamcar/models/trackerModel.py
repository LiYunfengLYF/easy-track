import torch
import torch.nn as nn
from .car_head import CARHead
from .resnet50 import resnet50
from ..config import cfg_siamcar
from .xcorr import xcorr_depthwise
from .adjustalllayer import AdjustAllLayer


class SiamCAR_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = cfg_siamcar()
        self.backbone = resnet50(used_layers=self.cfg.backbone_used_layers)
        self.neck = AdjustAllLayer(in_channels=self.cfg.adjust_in_channels,
                                   out_channels=self.cfg.adjust_out_channels)
        self.car_head = CARHead(self.cfg, 256)

        self.xcorr_depthwise = xcorr_depthwise
        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        self.zf = self.neck(self.backbone(z)) if self.cfg.adjust else self.backbone(z)

    def track(self, x):
        xf = self.neck(self.backbone(x)) if self.cfg.adjust else self.backbone(x)

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {'cls': cls, 'loc': loc, 'cen': cen}
