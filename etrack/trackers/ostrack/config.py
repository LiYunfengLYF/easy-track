class cfg_256:
    def __init__(self):
        # data
        self.search_size = 256
        self.search_factor = 4.0
        self.template_size = 128
        self.template_factor = 2.0

        # backbone
        self.backbone_type = r'vit_base_patch16_224_ce'
        self.drop_path_rate = 0.1
        self.ce_loc = [3, 6, 9]
        self.ce_keep_ratio = [0.7, 0.7, 0.7]
        self.stride = 16
        self.cat_mode = 'direct'
        self.return_inter = False
        self.return_stages = []
        self.sep_seg = False
        self.ce_template_range = r'CTR_POINT'

        # head
        self.head_type = r'CENTER'
        self.num_channels = 256


class cfg_384:
    def __init__(self):
        # data
        self.search_size = 384
        self.search_factor = 5.0
        self.template_size = 192
        self.template_factor = 2.0

        # backbone
        self.backbone_type = r'vit_base_patch16_224_ce'
        self.drop_path_rate = 0.1
        self.ce_loc = [3, 6, 9]
        self.ce_keep_ratio = [0.7, 0.7, 0.7]
        self.ce_template_range = r'CTR_POINT'
        self.stride = 16
        self.cat_mode = 'direct'
        self.return_inter = False
        self.return_stages = []
        self.sep_seg = False

        # head
        self.head_type = r'CENTER'
        self.num_channels = 256
