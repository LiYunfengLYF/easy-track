import cv2
import logging
import numpy as np

from ..utils import imread, draw_box, close_cv2_window, selectROI


def img_board(plt, img, label=None, num=None):
    plt.imshow(img)
    plt.axis('off')
    if label is not None:
        label = label + '-' + str(num) if num is not None else label
        plt.title(label)


def run_img_on_board(img, num, gt=None):
    image = imread(img)
    if gt is not None:
        image = draw_box(image, gt[num], color=(0, 0, 255), thickness=2)

    return image


def info_board(num, fps=None, iou=None):
    img = np.zeros((256, 256), np.uint8)
    img.fill(255)

    img = cv2.putText(img, f'Frame : {num}', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    if fps is not None:
        img = cv2.putText(img, f'FPS : {num}', (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    if iou is not None:
        img = cv2.putText(img, f'IOU : {num}', (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img


def run_tracker_on_board(tracker, img, num, start_num=0, select_roi=False, gt=None):
    if img is str:
        image = imread(img)
    else:
        image = img

    if num == start_num:
        if select_roi or gt is None:
            init_box = result = selectROI('SelectROI', image)
            close_cv2_window('SelectROI')
        else:
            init_box = result = gt[0]

        tracker.init(image, init_box)
    else:
        result = tracker.track(image)

    if result is not None:
        image = draw_box(image, result, color=(0, 0, 255), thickness=2)
    else:
        logging.warning(f'tracker result in {img} is none !!!')
        image = image

    return image, result


def curve_board(plt, x, y, label=None):
    plt.plot(x, y)
    if label is not None:
        plt.title(label)


def run_curve_on_board(result, gt_box, mode='precision'):
    assert mode in ['precision', 'norm precision', 'success rate']
    if mode == 'precision':
        pass

    return


def update_board(plt, pause):
    plt.draw()
    plt.pause(pause)
    plt.clf()
