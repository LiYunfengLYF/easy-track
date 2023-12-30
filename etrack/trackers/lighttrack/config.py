import torch


class Config():
    def __init__(self):
        self.penalty_k = 0.007
        self.window_influence = 0.225
        self.lr = 0.616
        self.windowing = 'cosine'

        self.exemplar_size = 127
        self.instance_size = 255
        self.total_stride = 16
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 1
        self.window_influence = 0.255

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.small_sz = 256
        self.big_sz = 288
