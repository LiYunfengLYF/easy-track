import os

import numpy as np
from tqdm import tqdm

from .analysis import calc_seq_performace
from ..utils import imread, draw_box, speed2waitkey, selectROI, txtread, seqread, imshow, \
    close_cv2_window, greenprint


def quick_start(tracker, seq_file, speed=20, imgs_type='.jpg'):
    """
    Description
        quick_start aim to help user to quickly observe the results of the tracker on an image sequence.
        It manually selects the initial bounding box and show results in each image by using blue bounding box.

    Params:
        tracker:
        seq_file:
        speed:      FPS speed, default = 20
        imgs_type:  image type, default = '.jpg'

    """
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


def run_sequence(tracker, seq_file, gt_file=None, save_path=None, save=False, visual=False, speed=20, imgs_type='.jpg',
                 select_roi=False, report_performance=True,desc=r'running:'):
    """
    Description


    Params:
        tracker:
        seq_file:
        speed:      FPS speed, default = 20
        imgs_type:  image type, default = '.jpg'
    """
    skip_circulate = False

    save_path = os.getcwd() if save_path is None else save_path

    select_roi = (True if gt_file is None else False) or select_roi

    imgs_list = seqread(seq_file, imgs_type=imgs_type) if isinstance(seq_file, str) else seq_file

    gt = txtread(gt_file) if isinstance(seq_file, str) else gt_file

    result_list = [] if save or report_performance else None
    try:
        winname = f'{tracker.name}-sequence'
    except:
        winname = 'sequence'

    if save:
        if os.path.exists(save_path):
            if len(txtread(save_path)) == len(imgs_list):
                greenprint(f'{save_path} is already!. skip it')
                skip_circulate = True
            else:
                os.remove(save_path)
                greenprint(f'{save_path} is already exit!. remove it')
    if not skip_circulate:
        for num, img_dir in tqdm(enumerate(imgs_list), total=len(imgs_list), desc=desc):
            image = imread(img_dir)

            if num == 0:
                if select_roi:
                    init_box = result = selectROI(winname, image)
                    close_cv2_window(winname) if visual is False else None
                else:
                    init_box = result = gt[0]

                tracker.init(image, init_box)
            else:
                result = tracker.track(image)

            if result is not None:
                if visual:
                    image = draw_box(image, result, color=(0, 0, 255), thickness=2)
                    imshow(winname, image, waitkey=speed2waitkey(speed))
                if save or report_performance:
                    result_list.append(result)
            else:
                raise print('result is None !!!')

    if result_list is not None and len(result_list) == len(imgs_list):
        results_boxes = np.array(result_list)
    else:
        results_boxes = txtread(save_path)

    if save:
        np.savetxt(fname=save_path, X=results_boxes, fmt='%d', delimiter='\t')

    if report_performance and (gt is not None):
        succ_score, prec_score, norm_prec_score = calc_seq_performace(results_boxes, gt)
        print(f'{tracker.name} performance:')
        print(f'\tSuccess Score:\t\t\t\t{round(succ_score, 2)}')
        print(f'\tPrecision Score:\t\t\t{round(prec_score, 2)}')
        print(f'\tNorm Precision Score:\t\t{round(norm_prec_score, 2)}')
        return succ_score, prec_score, norm_prec_score
