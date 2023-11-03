import os
import cv2
import numpy as np


def imread(filename):
    """
    Description
        imread is an extension of cv2.imread, which returns RGB images

    Params:
        filename:   the path of image

    Return:
        image:      np.array

    """
    try:
        image = cv2.imread(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        raise print(f'{filename} is wrong')


def txtread(filename, delimiter=None, dtype=np.float64):
    """
    Description
        txtread is an extension of np.loadtxt, support ',' and '\t' delimiter.
        The original implementation method is in the pytracking library at https://github.com/visionml/pytracking

    Params:
        filename:           the path of txt
        delimiter:          default is [',','\t']
        dtype:              default is np.float64

    Return:
        ground_truth_rect:  np.array(n,4), n is length of results

    """

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
    """
    Description
        Seqread reads all image items in the file and sorts them by numerical name
        It returns a list containing the absolute addresses of the images

        Sorting only supports two types, '*/1.jpg' and '*/*_1.jpg'

    Params:
        file:       images' file
        imgs_type:  default is '.jpg'

    Return:
        List of absolute paths of sorted images

    """

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
    """
    Description
        img_filter retains items in the specified format in the input list

    Params:
        imgs_list:          List of image path
        extension_filter:   default is '.jpg'

    Return:
        List of images path  with extension

    """
    return list(filter(lambda file: file.endswith(extension_filter), imgs_list))
