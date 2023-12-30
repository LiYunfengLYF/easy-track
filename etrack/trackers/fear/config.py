class cfg:
    def __init__(self):
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.windowing = 'cosine'
        self.total_stride = 16
        self.score_size = 16
        self.ratio = 0.94
        self.stride = 16
        self.bbox_ratio = 0.5
        self.template_bbox_offset = 0.2
        self.search_context = 2
        self.instance_size = 256
        self.template_size = 128
