class cfg_s50:
    def __init__(self):
        self.search_size = 320
        self.search_factor = 5.0

        self.template_size = 128
        self.template_factor = 2.0

        # backbone
        self.backbone_type = r'resnet50'
        self.backbone_multiplier = 0.1
        self.freeze_backbone_bn = True
        self.dilation = False
        self.predict_mask = False

        # position embedding
        self.position_embedding = r'sine'

        # transformer
        self.transformer_dropout = 0.1
        self.hidden_dim = 256
        self.num_heads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.divide_norm = False

        self.num_object_queries = 1
        self.deep_supervision = False

        # head
        self.head_type = r'CORNER'

class cfg_st50:
    def __init__(self):
        self.search_size = 320
        self.search_factor = 5.0

        self.template_size = 128
        self.template_factor = 2.0

        # backbone
        self.backbone_type = r'resnet50'
        self.backbone_multiplier = 0.1
        self.freeze_backbone_bn = True
        self.dilation = False
        self.predict_mask = False

        # position embedding
        self.position_embedding = r'sine'

        # transformer
        self.transformer_dropout = 0.1
        self.hidden_dim = 256
        self.num_heads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.divide_norm = False

        self.num_object_queries = 1
        self.deep_supervision = False

        # head
        self.head_type = r'CORNER'
        self.num_layer_head =3

class cfg_st101:
    def __init__(self):
        self.search_size = 320
        self.search_factor = 5.0

        self.template_size = 128
        self.template_factor = 2.0

        # backbone
        self.backbone_type = r'resnet101'
        self.backbone_multiplier = 0.1
        self.freeze_backbone_bn = True
        self.dilation = False
        self.predict_mask = False

        # position embedding
        self.position_embedding = r'sine'

        # transformer
        self.transformer_dropout = 0.1
        self.hidden_dim = 256
        self.num_heads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.divide_norm = False

        self.num_object_queries = 1
        self.deep_supervision = False

        # head
        self.head_type = r'CORNER'
        self.num_layer_head =3