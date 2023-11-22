class cfg_n4():
    def __init__(self):
        #
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]


        self.search_area_factor = 4.0
        self.template_area_factor = 2.0
        self.search_feature_sz = 32
        self.template_feature_sz = 16

        self.search_sz =self.search_feature_sz * 8
        self.temp_sz = self.template_feature_sz * 8

        self.center_jitter_factor = {'search': 3, 'template': 0}
        self.scale_jitter_factor = {'search': 0.25, 'template': 0}

        # Transformer
        self.position_embedding = 'sine'
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.featurefusion_layers = 4

        # tracking
        self.track_exemplar_size = 128
        self.track_window_influence = 0.49
        self.track_instance_size= 256