class cfg:
    def __init__(self):
        super().__init__()
        self.response_up = 16
        self.response_sz = 17

        self.scale_step = 1.0375
        self.scale_num = 3
        self.scale_penalty = 0.9745
        self.context = 0.5

        self.instance_sz = 255
        self.exemplar_sz = 127

        self.window_influence = 0.176
        self.total_stride = 8
        self.scale_lr = 0.59
