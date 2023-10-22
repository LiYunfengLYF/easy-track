import os
from visual.load import txtread


def decode_img_file(file, imgs_type='.jpg'):
    if type(file) is str:
        return imfile_str2list(file, imgs_type)
    elif type(file) in [list, tuple]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def decode_txt_file(file):
    if type(file) is str:
        return txtread(file)
    elif type(file) in [list, tuple]:
        return file
    else:
        raise print(f'Unknown file type: {type(file)}')


def imfile_str2list(file, imgs_type='.jpg'):
    output_list = sorted(img_filter(os.listdir(file), imgs_type), key=lambda x: int(x.split('.')[-2]))
    return [os.path.join(file, item) for item in output_list]


def check_is_img(item):
    if type(item) is str:
        return item.split('.')[-1] in ['jpg', 'jpeg', 'png', 'tif', 'bmp']
    else:
        return False


def img_filter(imgs_list, extension_filter=r'.jpg'):
    return list(filter(lambda file: file.endswith(extension_filter), imgs_list))


def speed2waitkey(speed):
    if speed == 0:
        return 0
    else:
        return int((1 / speed) * 1000)
