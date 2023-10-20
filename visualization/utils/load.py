import cv2
import numpy as np


def imread(filename):
    try:
        image = cv2.imread(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        raise print(f'{filename} is wrong')


def txtread(filename, delimiter=None, dtype=np.float64):
    if delimiter is None:
        delimiter = [',', '\t']

    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(filename, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(filename))
    else:
        ground_truth_rect = np.loadtxt(filename, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect
