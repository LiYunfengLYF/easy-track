import torch
from etrack import trans_checkpoint_keys, show_checkpoint_keys

checkpoint_path = r'baseline_R101/STARKST_ep0050.pth.tar'

trans_checkpoint_keys(checkpoint_path, name=r'starkst101')
