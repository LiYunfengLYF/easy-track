class cfg_siamrpn_alex_dwcorr:
    def __init__(self):
        self.backbone_type = 'alexnetlegacy'
        self.backbone_width_mult = 1.0

        self.adjust = False

        self.rpn_head_type = 'DepthwiseRPN'
        self.rpn_anchor_num = 5
        self.rpn_in_channels = 256
        self.rpn_out_channels = 256

        self.anchor_stride = 8
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        self.anchor_scales = [8]
        self.anchor_num = 5

        self.track_exemplar_size = 127
        self.track_instance_size = 287
        self.track_window_influence = 0.4
        self.track_penalty_k = 0.16
        self.track_base_size = 0
        self.track_context_amount = 0.5
        self.track_lr = 0.3


class cfg_siamrpnpp_mobilev2_dwcorr:
    def __init__(self):
        self.backbone_type = 'mobilenetv2'
        self.backbone_used_layers = [3, 5, 7]
        self.backbone_width_mult = 1.4

        self.adjust = True
        self.adjust_type = 'AdjustAllLayer'
        self.adjust_in_channels = [44, 134, 448]
        self.adjust_out_channels = [256, 256, 256]

        self.rpn_head_type = 'MultiRPN'
        self.rpn_anchor_num = 5
        self.rpn_in_channels = [256, 256, 256]
        self.rpn_weighted = False

        self.anchor_stride = 8
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        self.anchor_scales = [8]
        self.anchor_num = 5

        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_window_influence = 0.4
        self.track_penalty_k = 0.04
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_lr = 0.5

class cfg_siamrpnpp_rensnet_dwcorr:
    def __init__(self):
        self.backbone_type = 'resnet50'
        self.backbone_used_layers = [2, 3, 4]

        self.adjust = True
        self.adjust_type = 'AdjustAllLayer'
        self.adjust_in_channels = [512, 1024, 2048]
        self.adjust_out_channels = [256, 256, 256]

        self.rpn_head_type = 'MultiRPN'
        self.rpn_anchor_num = 5
        self.rpn_in_channels = [256, 256, 256]
        self.rpn_weighted = True

        self.anchor_stride = 8
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        self.anchor_scales = [8]
        self.anchor_num = 5

        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_window_influence = 0.42
        self.track_penalty_k = 0.05
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_lr = 0.38