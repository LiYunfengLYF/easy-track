import torch.nn as nn
from trackers.siamfc.models.backbone import AlexNetV1
from trackers.siamfc.models.xcorr import xcorr


class SiamFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AlexNetV1()
        self.head = xcorr(out_scale=0.001)

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

    def forward_backbone(self, z):
        z = self.backbone(z)
        return z

    def forward_tracking(self, z_feat, x):
        x = self.backbone(x)
        return self.head(z_feat, x)


class SiamFC_deploy(SiamFC):

    def forward(self, zf, x):
        x = self.backbone(x)
        return self.head(zf, x)
