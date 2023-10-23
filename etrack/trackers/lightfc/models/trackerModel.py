import torch.nn as nn
from .backbone import MobileNetV2
from .fusion import PWCorr_SE_SCF31_IAB11_Concat_Release
from .head import RepN33_SE_Center_Concat


class LightFC(nn.Module):
    def __init__(self):
        super(LightFC, self).__init__()

        self.backbone = MobileNetV2()

        self.fusion = PWCorr_SE_SCF31_IAB11_Concat_Release(num_kernel=64,
                                                           adj_channel=96)

        self.head = RepN33_SE_Center_Concat(inplanes=192,
                                            channel=256,
                                            feat_sz=16,
                                            stride=16,
                                            freeze_bn=False, )

    def forward(self, z, x, train=False):
        z = self.backbone(z)
        x = self.backbone(x)
        opt = self.fusion(z, x)
        out = self.head(opt)
        return out
    def forward_backbone(self, z):
        z = self.backbone(z)
        return z

    def forward_tracking(self, z_feat, x):
        x = self.backbone(x)
        opt = self.fusion(z_feat, x)
        out = self.head(opt)
        return out
