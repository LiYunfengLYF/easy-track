import os
import sys


def no_print(func):
    def wrapper(*args, **kargs):
        sys.stdout = open(os.devnull, 'w')
        f = func(*args, **kargs)
        sys.stdout = sys.__stdout__
        return f

    return wrapper


def tqdm_update(bar, seq_id, length, seq_name):
    bar.set_description(f'[{seq_id + 1}/{length}] {seq_name} ')
    bar.update(1)


def check_is_img(item):
    if type(item) is str:
        return item.split('.')[-1] in ['jpg', 'jpeg', 'png', 'tif', 'bmp']
    else:
        return False
