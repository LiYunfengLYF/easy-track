import torch.nn as nn
from ..config import cfg_siamban
from .resnet import resnet50
from .adjustalllayer import AdjustAllLayer
from .ban_head import MultiBAN
from .ban_acm import MultiBAN_ACM


class SiamBAN_Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfg = cfg_siamban()
        self.backbone = resnet50(used_layers=self.cfg.backbone_used_layers)

        self.neck = AdjustAllLayer(in_channels=self.cfg.adjust_in_channels,
                                   out_channels=self.cfg.adjust_out_channels)

        self.head = MultiBAN(in_channels=self.cfg.ban_in_channels,
                             cls_out_channels=self.cfg.ban_cls_out_channels,
                             weighted=self.cfg.ban_weighted)

    def template(self, z):
        self.zf = self.neck(self.backbone(z)) if self.cfg.adjust else self.backbone(z)

    def track(self, x):
        xf = self.neck(self.backbone(x)) if self.cfg.adjust else self.backbone(x)
        cls, loc = self.head(self.zf, xf)
        return {'cls': cls, 'loc': loc}


class SiamBAN_ACM_Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfg = cfg_siamban()
        self.backbone = resnet50(used_layers=self.cfg.backbone_used_layers)

        self.neck = AdjustAllLayer(in_channels=self.cfg.adjust_in_channels,
                                   out_channels=self.cfg.adjust_out_channels)

        self.head = MultiBAN_ACM(in_channels=self.cfg.ban_in_channels,
                                 cls_out_channels=self.cfg.ban_cls_out_channels,
                                 weighted=self.cfg.ban_weighted)

    def template(self, z, bbox):
        self.bbox = bbox
        self.zf = self.neck(self.backbone(z)) if self.cfg.adjust else self.backbone(z)
        self.head.init(self.zf, bbox)

    def track(self, x):
        xf = self.neck(self.backbone(x)) if self.cfg.adjust else self.backbone(x)
        cls, loc = self.head.track(xf)
        return {'cls': cls, 'loc': loc}
