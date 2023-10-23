import os
import torch


class Tracker:
    def __init__(self, checkpoint_path, use_cuda, name):
        self.network = None
        self.name = name
        self.checkpoint = os.path.join(os.getcwd(), 'etrack_checkpoints') if checkpoint_path is None else checkpoint_path

        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def init(self, image, info):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError

    def load_checkpoint(self):
        self.checkpoint = os.path.join(self.checkpoint, self.name + '.pth')
        self.network.load_state_dict(torch.load(self.checkpoint), strict=True)
