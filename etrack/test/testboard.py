import logging
from ..utils import imread, draw_box, silentSelectROI, close_cv2_window
from .analysis import calc_precision, calc_iou


def img_board(plt, img, label=None):
    plt.imshow(img)
    plt.axis('off')
    if label is not None:
        plt.title(label)


def run_img_on_board(img, num, gt=None):
    image = imread(img)
    if gt is not None:
        image = draw_box(image, gt[num], color=(0, 0, 255), thickness=2)

    return image


def run_tracker_on_board(tracker, img, num, start_num=0, select_roi=False, gt=None):
    image = imread(img)
    if num == start_num:
        if select_roi or gt is None:
            init_box = result = silentSelectROI('SelectROI', image)
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
