import cv2
from tqdm import tqdm
from ..utils import draw_box, tqdm_update, imread, load_seq_result, decode_img_file, decode_txt_file, speed2waitkey

from ..utils.dataset import otbDataset, utb180Dataset, uot100Dataset, lasotDataset


def imshow(winname, image, waitkey=0):
    """
    Description
        imshow is an extension of cv2.imshow
        Different with cv2.imshow, it input is an RGB image, and window size is variable

    Params:
        winname:    str
        image:      np.array
        waitkey:    default is 0

    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)


def seqshow(imgs_file, imgs_type=r'jpg', result_file=None, gt_file=None, show_gt=True, speed=20, tracker_name='',
            seq_name=r'default', result_color=(0, 0, 255), thickness=2):
    """
    Description
        seqshow visualizes the bounding box results of the tracker in image sequence

        if results_file is none, tracker results (default is red bounding box) will not be displayed on sequence
        if gt_file is none or show_gt is False, groundtruth (green bounding box) will not be displayed on sequence

    Params:
        imgs_file:      str
        imgs_type:      default is '.jpg', you can change it
        result_file:    str
        gt_file:        str
        show_gt:        True or False
        speed:          FPS
        tracker_name:   str
        seq_name:       str
        result_color:   default is red (0,0,255), you can change it
        thickness:      int

    """

    # decode images, tracker's results and gt
    imgs_list = decode_img_file(imgs_file, imgs_type)
    result = decode_txt_file(result_file) if result_file is not None else None
    gt = decode_txt_file(gt_file) if gt_file is not None else None

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


def show_lasot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
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


def show_uot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
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


def show_utb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    """
    Description
        show_utb visualizes the bounding box results of the tracker in uot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on utb
        if show_gt is False, groundtruth (green bounding box) will not be displayed on utb

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str

    """
    utb = utb180Dataset(dataset_files)
    datasetshow(utb, result_file=result_file, show_gt=show_gt, speed=speed, tracker_name=tracker_name,
                result_color=result_color)


def close_cv2_window(winname):
    """
    Description
        close an opened window of cv2

    Params:
        winname: str

    """
    cv2.destroyWindow(winname)
