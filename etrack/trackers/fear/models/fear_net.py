from typing import Dict, Tuple

import torch
import torch.nn as nn

from .blocks import Encoder, AdjustLayer, BoxTower
from ..utils import make_grid


class FEARModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FEARNet()

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]]


class FEARNet(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            pretrained: bool = True,
            score_size: int = 25,
            adjust_channels: int = 256,
            total_stride: int = 8,
            instance_size: int = 255,
            towernum: int = 2,
            max_layer: int = 4,
            crop_template_features: bool = True,
            conv_block: str = "sep_conv",
            mobile: bool = True,
    ) -> None:
        max_layer2name = {3: "layer2", 4: "layer1"}
        assert max_layer in max_layer2name

        super().__init__()
        self.encoder = Encoder(pretrained)
        self.neck = AdjustLayer(
            in_channels=self.encoder.encoder_channels[max_layer2name[max_layer]], out_channels=adjust_channels
        )
        self.connect_model = BoxTower(
            inchannels=adjust_channels,
            outchannels=adjust_channels,
            towernum=towernum,
            conv_block=conv_block,
            mobile=mobile,
        )
        self.search_size = img_size
        self.score_size = score_size
        self.total_stride = total_stride
        self.instance_size = instance_size
        self.size = 1
        self.max_layer = max_layer
        self.crop_template_features = crop_template_features
        self.grid_x = torch.empty(0)
        self.grid_y = torch.empty(0)
        self.features = None
        self.grids(self.size)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.encoder.stages[: self.max_layer]:
            x = stage(x)
        return x

    def get_features(self, crop: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(crop)
        features = self.neck(features)
        return features

    def grids(self, size: int) -> None:
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        grid_x, grid_y = make_grid(self.score_size, self.total_stride, self.instance_size)
        self.grid_x, self.grid_y = grid_x.unsqueeze(0).repeat(size, 1, 1, 1), grid_y.unsqueeze(0).repeat(size, 1, 1, 1)

    def connector(self, template_features: torch.Tensor, search_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        bbox_pred, cls_pred, _, _ = self.connect_model(search_features, template_features)
        return {
            'TARGET_REGRESSION_LABEL_KEY': bbox_pred,
            'TARGET_CLASSIFICATION_KEY': cls_pred,
        }

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        template, search = x
        self.size = search.size(0)
        template_features = self.get_features(template)
        search_features = self.get_features(search)
        return self.connector(template_features=template_features, search_features=search_features)

    def track(
            self,
            search: torch.Tensor,
            template_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        search_features = self.get_features(search)
        return self.connector(template_features=template_features, search_features=search_features)
