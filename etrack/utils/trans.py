import os
import cv2
import sys
from .load import txtread, seqread
from .decorator import no_print,no_print_cv2


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
    if speed == 0:
        return 0
    else:
        return int((1 / speed) * 1000)


def selectROI(winname, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = cv2.selectROI(windowName=winname, img=img)
    return bbox


@no_print_cv2
def silentSelectROI(winname, img):
    return selectROI(winname, img)
