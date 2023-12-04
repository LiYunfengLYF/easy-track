import torch.nn as nn
from .backbone import AlexNetLegacy, MobileNetV2,resnet50
from .neck import AdjustAllLayer
from .head import DepthwiseRPN, MultiRPN
from ..config import cfg_siamrpn_alex_dwcorr, cfg_siamrpnpp_mobilev2_dwcorr, cfg_siamrpnpp_rensnet_dwcorr


class SianRPN_Alex(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = cfg_siamrpn_alex_dwcorr()
        self.backbone = AlexNetLegacy(width_mult=self.cfg.backbone_width_mult)
        self.rpn_head = DepthwiseRPN(anchor_num=self.cfg.anchor_num,
                                     in_channels=self.cfg.rpn_in_channels,
                                     out_channels=self.cfg.rpn_out_channels, )
        self.neck = None

    def template(self, z):
        self.zf = self.neck(self.backbone(z)) if self.cfg.adjust else self.backbone(z)

    def track(self, x):
        xf = self.neck(self.backbone(x)) if self.cfg.adjust else self.backbone(x)
        cls, loc = self.rpn_head(self.zf, xf)
        return {'cls': cls, 'loc': loc, }


class SiamRPNpp_Mobilev2(SianRPN_Alex):
    def __init__(self):
        super().__init__()
        self.cfg = cfg_siamrpnpp_mobilev2_dwcorr()
        self.backbone = MobileNetV2(used_layers=self.cfg.backbone_used_layers,
                                    width_mult=self.cfg.backbone_width_mult)
        self.neck = AdjustAllLayer(in_channels=self.cfg.adjust_in_channels,
                                   out_channels=self.cfg.adjust_out_channels)

        self.rpn_head = MultiRPN(self.cfg.rpn_anchor_num,
                                 self.cfg.rpn_in_channels,
                                 self.cfg.rpn_weighted)


class SiamRPNpp_Resnet50(SianRPN_Alex):
    def __init__(self):
        super().__init__()
        self.cfg = cfg_siamrpnpp_rensnet_dwcorr()
        self.backbone = resnet50(used_layers=self.cfg.backbone_used_layers, )
        self.neck = AdjustAllLayer(in_channels=self.cfg.adjust_in_channels,
                                   out_channels=self.cfg.adjust_out_channels)

        self.rpn_head = MultiRPN(self.cfg.rpn_anchor_num,
                                 self.cfg.rpn_in_channels,
                                 self.cfg.rpn_weighted)
