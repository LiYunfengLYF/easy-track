import os

import cv2
from tqdm import tqdm
from ..visual import imshow, close_cv2_window
from ..utils import imread, draw_box, speed2waitkey, selectROI, silentSelectROI, txtread, seqread


def quick_start(tracker, seq_file, speed=20, imgs_type='jpg'):
    imgs_list = seqread(seq_file, imgs_type=imgs_type)

    for num, img_dir in enumerate(imgs_list):
        image = imread(img_dir)

        if num == 0:
            init_box = result = selectROI(r'quick_start', image)

            tracker.init(image, init_box)

        else:
            result = tracker.track(image)

        if result is not None:
            image = draw_box(image, result, color=(0, 0, 255), thickness=2)

        imshow('quick_start', image, waitkey=speed2waitkey(speed))


def run_sequence(tracker, seq_file, gt_file=None, save_path=None, save=False, visual=False, speed=20, imgs_type='jpg',
                 select_roi=False):
    save_path = os.getcwd() if save_path is not None else save_path
    select_roi = (True if gt_file is not None else False) or select_roi

    imgs_list = seqread(seq_file, imgs_type=imgs_type)
    gt = txtread(gt_file) if gt_file is not None else None

    result_list = [] if save else None

    for num, img_dir in tqdm(enumerate(imgs_list), total=len(imgs_list), desc=r'running:'):
        image = imread(img_dir)

        if num == 0:
            if select_roi:
                init_box = result = silentSelectROI(r'sequence', image)
                close_cv2_window(r'sequence') if visual is False else None
            else:
                init_box = result = gt[0]

            tracker.init(image, init_box)
        else:
            result = tracker.track(image)

        if result is not None:
            if visual:
                image = draw_box(image, result, color=(0, 0, 255), thickness=2)
                imshow('sequence', image, waitkey=speed2waitkey(speed))
            if save:
                result_list.append(result)
        else:
            raise print('result is None !!!')
    if save:
        pass
