import os
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


def seqread(file, imgs_type='.jpg'):
    try:
        output_list = sorted(img_filter(os.listdir(file), imgs_type), key=lambda x: int(x.split('.')[-2]))
    except ValueError:
        output_list = sorted(img_filter(os.listdir(file), imgs_type),
                             key=lambda x: int(x.split('.')[-2].split('_')[-1]))

    return [os.path.join(file, item) for item in output_list]


def load_seq_result(dataset_file, seq_name):
    seq_result_path = os.path.join(dataset_file, seq_name + '.txt')
    result = txtread(seq_result_path, [',', '\t'])
    return result


def img_filter(imgs_list, extension_filter=r'.jpg'):
    return list(filter(lambda file: file.endswith(extension_filter), imgs_list))
