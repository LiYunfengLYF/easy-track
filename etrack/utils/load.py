import os
import cv2
import json
import pickle
import numpy as np


def imread(filename: str) -> np.array:
    """
    Description
        imread is an easy extension of cv2.imread, which returns RGB images
    """
    try:
        image = cv2.imread(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        raise print(f'{filename} is wrong')


def imwrite(filename: str, image: np.array):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


def txtread(filename: str, delimiter: [str, list] = None) -> np.ndarray:
    """
    Description
        txtread is an extension of np.loadtxt, support ',' and '\t' delimiter.
        The original implementation method is in the pytracking library at https://github.com/visionml/pytracking
    """

    if delimiter is None:
        delimiter = [',', '\t']

    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(filename, delimiter=d, dtype=np.float64)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(filename))
    else:
        ground_truth_rect = np.loadtxt(filename, delimiter=delimiter, dtype=np.float64)
        return ground_truth_rect


def easy_txtread(file: [str, list, tuple]):
    if type(file) is str:
        return txtread(file)
    elif type(file) in [list, tuple]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def seqread(file: str, imgs_type='.jpg'):
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


def easy_seqread(file: [str, list, tuple, np.ndarray], imgs_type: str = '.jpg'):
    if type(file) is str:
        return seqread(file, imgs_type)
    elif type(file) in [list, tuple, np.ndarray]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def pklread(file: str) -> dict:
    """
    Description
        pklread is an easy extension of pickle.load
    """
    if os.path.exists(file):
        f_read = open(file, 'rb')
        pkl = pickle.load(f_read)
        f_read.close()
    else:
        raise f'{file} is not exit!!!'
    return pkl


def pklwrite(file: str, pkl: dict):
    """
    Description
        pklwrite is an easy extension of pickle.dump
    """
    f_write = open(file, 'wb')
    pickle.dump(pkl, f_write)
    f_write.close()


def jsonread(file: str) -> dict:
    """
    Description
        jsonread is an easy extension of json.loads
    """
    f = open(file, 'r')
    content = f.read()
    raw_meta = json.loads(content)
    f.close()
    return raw_meta


def load_seq_result(dataset_file: str, seq_name: str) -> np.ndarray:
    seq_result_path = os.path.join(dataset_file, seq_name + '.txt')
    result = txtread(seq_result_path, [',', '\t'])
    return result


def img_filter(imgs_list: [str, list], extension_filter: str = r'.jpg') -> list:
    """
    Description
        img_filter retains items in the specified format in the input list
    Params:
        extension_filter:   default is '.jpg'
    """
    return list(filter(lambda file: file.endswith(extension_filter), imgs_list))
