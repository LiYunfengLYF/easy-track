class cfg_siamban:
    def __init__(self):
        self.backbone_type = 'resnet50'
        self.backbone_used_layers = [2, 3, 4]

        self.adjust = True
        self.adjust_type = 'AdjustAllLayer'
        self.adjust_in_channels = [512, 1024, 2048]
        self.adjust_out_channels = [256, 256, 256]

        self.ban_in_channels = [256, 256, 256]
        self.ban_cls_out_channels = 2
        self.ban_weighted = True

        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_penalty_k = 0.08513642556896711
        self.track_window_influence = 0.4632532824922313
        self.track_lr = 0.44418184746462425

        self.point_stride = 8

class cfg_siamban_acm:
    def __init__(self):
        self.backbone_type = 'resnet50'
        self.backbone_used_layers = [2, 3, 4]

        self.adjust = True
        self.adjust_type = 'AdjustAllLayer'
        self.adjust_in_channels = [512, 1024, 2048]
        self.adjust_out_channels = [256, 256, 256]

        self.ban_in_channels = [256, 256, 256]
        self.ban_cls_out_channels = 2
        self.ban_weighted = True

        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_penalty_k = 0.08513642556896711
        self.track_window_influence = 0.4632532824922313
        self.track_lr = 0.44418184746462425

        self.point_stride = 8
