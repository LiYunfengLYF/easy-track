class cfg_siamcar:
    def __init__(self):
        self.backbone_type = 'resnet50'
        self.backbone_used_layers = [2, 3, 4]

        self.adjust = True
        self.adjust_type = 'AdjustAllLayer'
        self.adjust_in_channels = [512, 1024, 2048]
        self.adjust_out_channels = [256, 256, 256]

        self.track_penalty_k = 0.04
        self.track_window_influence = 0.44
        self.track_lr = 0.33
        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_context_amount = 0.5
        self.track_stride = 8
        self.track_score_size = 25
        self.track_hanning = True
        self.track_region_s = 0.1
        self.track_region_l = 0.44

        self.train_num_classes = 2
        self.train_num_convs = 4
        self.train_prior_prob = 0.01

        self.h_lr = 0.3
        self.h_penalty_k = 0.4
        self.h_window_lr = 0.4
