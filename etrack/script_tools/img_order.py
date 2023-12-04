import os
import cv2
import shutil
import logging
from tqdm import tqdm
from ..utils import seqread


def trans_imgs_order_name(file, save_file, sort=True, imgs_format='.jpg', preread=True, format_name=False, width=4,
                          start=1, end=None, ) -> None:
    """
    Description
        transfer image into an order name, and save in save_file
        if sort is False, it will directly read images. The original order of imgs may not be preserved
        if preread is False, it will directly copy and paste, image will not be opened
        if format_name is True, images' name is like 0001.jpg, 0002.jpg (width=4), ... else 1.jpg, 2.jpg, ...

    Params:
        file:           str
        save_file:      str
        sort:           True or False
        imgs_format:    str, default is '.jpg'
        preread:        True or False
        format_name:    True or False
        width:          int
        start:          int
        end:            int

    """
    assert imgs_format in ['.jpg', '.png', '.jpeg']

    if not os.path.isdir(file):
        raise 'Input file is not a dir, please check it !!!'

    if not os.path.exists(save_file):
        os.makedirs(save_file)

    if sort:
        file_items = seqread(file, imgs_format)
        if len(file_items) == 0:
            logging.warning(f'There is no images with {imgs_format}, please check it or set sort=False')
        elif len(file_items) != len(os.listdir(file)):
            logging.warning(f'There may be different format of images, please check it')
    else:
        file_items = [os.path.join(file, item) for item in os.listdir(file)]
    end = len(file_items) if end is None else end

    if format_name:
        name_list = [f'{i:0{width}}' + imgs_format for i in range(start, end + 1)]
    else:
        name_list = [str(i) + imgs_format for i in range(start, end + 1)]

    save_items = [os.path.join(save_file, item) for item in name_list]
    if not preread:
        logging.warning('Images are converted directly and will not be opened ！！！')
        logging.warning('It is recommended to set: preread = True')

    for i in tqdm(range(end + 1 - start), total=(end + 1 - start), desc='running: '):
        if preread:
            try:
                images = cv2.imread(file_items[i])
                cv2.imwrite(save_items[i], images)
            except Exception as E:
                print(E)
                raise f'Error at item {file_items[i]}, please check it !!!'
        else:
            shutil.copy(file_items[i], save_items[i])

    print(f"Finish trans image from {file} to {save_file}")
