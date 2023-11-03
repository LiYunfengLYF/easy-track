import os
import cv2
import sys

import numpy as np
import torch

from .load import txtread, seqread
from .decorator import no_print, no_print_cv2


def decode_img_file(file, imgs_type='.jpg'):
    if type(file) is str:
        return seqread(file, imgs_type)
    elif type(file) in [list, tuple]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def decode_txt_file(file):
    if type(file) is str:
        return txtread(file)
    elif type(file) in [list, tuple]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def check_is_img(item):
    if type(item) is str:
        return item.split('.')[-1] in ['jpg', 'jpeg', 'png', 'tif', 'bmp']
    else:
        return False


def speed2waitkey(speed):
    """
    Description
        trans fps to waitkey of cv2

    Params:
        speed:      fps, int

    return:
        waitkey:    int

    """
    if speed == 0:
        return 0
    else:
        return int((1 / speed) * 1000)


def selectROI(winname, img):
    """
    Description
        selectROI is an extension of cv2.selectROI
        input image is RGB rather BGR

    Params:
        winname:    name
        img:        np.array

    return:
        bbox:       [x,y,w,h]

    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = cv2.selectROI(windowName=winname, img=img)
    return bbox


@no_print_cv2
def silentSelectROI(winname, img):
    return selectROI(winname, img)


def img2tensor(img, device='cuda:0'):
    """
    Description
        transfer an img to a tensor
        mean: [0.485, 0.456, 0.406]
        std:  [0.229, 0.224, 0.225]

    Params:
        img:       np.array
        device:    default is 'cuda:0'

    return:
        Tensor:    torch.tensor(1,3,H,W)

    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    img_tensor = torch.tensor(img).to(device).float().permute((2, 0, 1)).unsqueeze(dim=0)
    return ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
