import cv2
from tqdm import tqdm
from ..utils import draw_box
from ..utils import tqdm_update
from ..utils import imread, load_seq_result
from ..utils import decode_img_file, decode_txt_file, speed2waitkey

from ..utils.dataset import otbDataset
from ..utils.dataset import utb180Dataset
from ..utils.dataset import uot100Dataset
from ..utils.dataset import lasotDataset


def imshow(winname, image, waitkey=0):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)


def seqshow(imgs_file, imgs_type=r'jpg', result_file=None, gt_file=None, show_gt=True, speed=20, tracker_name='',
            seq_name=r'default', result_color=(0, 0, 255), thickness=2):
    # decode images, tracker's results and gt
    imgs_list = decode_img_file(imgs_file, imgs_type)
    result = decode_txt_file(result_file) if result_file is not None else None
    gt = decode_txt_file(gt_file) if gt_file is not None else None

    #
    show_name = tracker_name + '-' + seq_name
    for num, img_dir in enumerate(imgs_list):

        image = imread(img_dir)

        if show_gt and gt_file:
            image = draw_box(image, gt[num], color=(0, 255, 0), thickness=thickness)

        if result is not None:
            image = draw_box(image, result[num], color=result_color, thickness=thickness)

        imshow(show_name, image, waitkey=speed2waitkey(speed))

    close_cv2_window(show_name)


def datasetshow(dataset, result_file=None, show_gt=True, thickness=2, speed=20, tracker_name=r'',
                result_color=(0, 0, 255), ):
    for seq_id, (seq_name, imgs_dir, gt) in enumerate(dataset):
        show_name = tracker_name + '-' + seq_name
        seq_result = load_seq_result(dataset_file=result_file, seq_name=seq_name)

        with tqdm(total=len(imgs_dir), desc=None) as bar:
            for num, img_dir in enumerate(imgs_dir):

                image = imread(img_dir)

                if show_gt:
                    image = draw_box(image, gt[num], color=(0, 255, 0), thickness=thickness)

                if result_file is not None:
                    image = draw_box(image, seq_result[num], color=result_color, thickness=thickness)

                imshow(show_name, image, waitkey=speed2waitkey(speed))

                tqdm_update(bar, seq_id, len(dataset), show_name)

        close_cv2_window(show_name)


def show_otb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    otb = otbDataset(dataset_files)
    datasetshow(otb, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_lasot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    lasot = lasotDataset(dataset_files)
    datasetshow(lasot, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_uot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    uot = uot100Dataset(dataset_files)
    datasetshow(uot, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_utb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    utb = utb180Dataset(dataset_files)
    datasetshow(utb, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def close_cv2_window(seq_name):
    cv2.destroyWindow(seq_name)
