import cv2
import numpy as np
from tqdm import tqdm
from .others import tqdm_update
from .load import easy_seqread, easy_txtread
from ..utils import draw_box, imread, load_seq_result, speed2waitkey
from ..utils.dataset import otbDataset, utb180Dataset, uot100Dataset, lasotDataset


def imshow(winname: str, image: np.array, waitkey=0, resize: [tuple, list] = None):
    """
    Description
        imshow is an easy extension of cv2.imshow
        Different with cv2.imshow, its input is an RGB image, and window size is variable
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)

    if resize is not None:
        cv2.resizeWindow(winname, resize)
    cv2.imshow(winname, image)
    key = cv2.waitKey(waitkey)
    return key


def seqshow(imgs_file: str, imgs_type: str = r'jpg', result_file: str = None, gt_file: str = None, show_gt=True,
            speed: int = 20, tracker_name: str = '', seq_name: int = r'default', result_color: tuple = (0, 0, 255),
            thickness: int = 2) -> None:
    """
    Description
        seqshow visualizes the bounding box results of the tracker in image sequence
        if results_file is none, tracker results (default is red bounding box) will not be displayed on sequence
        if gt_file is none or show_gt is False, groundtruth (green bounding box) will not be displayed on sequence
    """

    # decode images, tracker's results and gt
    imgs_list = easy_seqread(imgs_file, imgs_type)
    result = easy_txtread(result_file) if result_file is not None else None
    gt = easy_txtread(gt_file) if gt_file is not None else None

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
                result_color=(0, 0, 255), ) -> None:
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


def show_otb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True,
             tracker_name=r'') -> None:
    """
    Description
        show_otb visualizes the bounding box results of the tracker in otb benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on OTB
        if show_gt is False, groundtruth (green bounding box) will not be displayed on OTB

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str

    """
    otb = otbDataset(dataset_files)
    datasetshow(otb, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_lasot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True,
               tracker_name=r'') -> None:
    """
    Description
        show_lasot visualizes the bounding box results of the tracker in lasot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on lasot
        if show_gt is False, groundtruth (green bounding box) will not be displayed on lasot

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str

    """
    lasot = lasotDataset(dataset_files)
    datasetshow(lasot, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_uot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True,
             tracker_name=r'') -> None:
    """
    Description
        show_uot visualizes the bounding box results of the tracker in uot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on uot
        if show_gt is False, groundtruth (green bounding box) will not be displayed on uot

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str

    """
    uot = uot100Dataset(dataset_files)
    datasetshow(uot, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def show_utb(dataset_files: str, result_file: str = None, speed: int = 20, result_color: tuple = (0, 0, 255),
             show_gt: bool = True, tracker_name: str = r'') -> None:
    """
    Description
        show_utb visualizes the bounding box results of the tracker in uot benchmark
        if results_file is none, tracker results (default is red bounding box) will not be displayed on utb
        if show_gt is False, groundtruth (green bounding box) will not be displayed on utb
    """
    utb = utb180Dataset(dataset_files)
    datasetshow(utb, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def close_cv2_window(winname: str) -> None:
    """
    Description
        close an opened window of cv2

    Params:
        winname: str

    """
    cv2.destroyWindow(winname)


def greenprint(text: str) -> None:
    print(f"\033[92m{text}\033[0m")
