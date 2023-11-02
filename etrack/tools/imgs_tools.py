import os
import cv2
import shutil
import logging

import torch
from tqdm import tqdm
from .utils.MobileNetV2 import mobilenet_v2
from ..utils import imread, seqread, img2tensor


def trans_imgs_name(file, save_file, sort=True, imgs_format='.jpg', preread=True, format_name=False, width=4,
                    start=1, end=None, ):
    assert imgs_format in ['.jpg', '.png', '.jpeg']

    if not os.path.isdir(file):
        raise 'Input file is not a dir, please check it !!!'

    if not os.path.exists(save_file):
        os.makedirs(save_file)

    if sort:
        file_items = seqread(file, imgs_format)
        if len(file_items) == 0:
            logging.warning(f'There is no images with {imgs_format}, please check it or set sort=False')
        elif len(file_items) != len(os.listdir(file)):
            logging.warning(f'There may be different format of images, please check it')
    else:
        file_items = [os.path.join(file, item) for item in os.listdir(file)]
    end = len(file_items) if end is None else end

    if format_name:
        name_list = [f'{i:0{width}}' + imgs_format for i in range(start, end + 1)]
    else:
        name_list = [str(i) + imgs_format for i in range(start, end + 1)]

    save_items = [os.path.join(save_file, item) for item in name_list]
    if not preread:
        logging.warning('Images are converted directly and may not be opened ！！！')
        logging.warning('It is recommended to set: preread = True')

    for i in tqdm(range(end + 1 - start), total=(end + 1 - start), desc='running: '):
        if preread:
            try:
                images = cv2.imread(file_items[i])
                cv2.imwrite(save_items[i], images)
            except Exception as E:
                print(E)
                raise f'Error at item {file_items[i]}, please check it !!!'
        else:
            shutil.copy(file_items[i], save_items[i])

    print(f"Finish trans image from {file} to {save_file}")


def remove_same_img(file, save_file, checkpoint_path=None, device='cuda:0', resize=(320, 640), thred=0.4, show_same=False):
    logging.warning('Use MobilenetV2 to compute the similarity of images')
    logging.warning('MobileNetV2 use pretrained model is mobilenet_v2-b0353104.pth, download it at torchvision toolkit')
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    else:
        shutil.rmtree(save_file)
        os.makedirs(save_file)
    model = mobilenet_v2()
    checkpoint_path = os.path.join(os.getcwd(), 'etrack_checkpoints',
                                   'mobilenet_v2-b0353104.pth') if checkpoint_path is None else checkpoint_path
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device).eval()

    try:
        imgs_list = seqread(file)
    except:
        raise "Input file has unsort name, please use function: trans_img_name to sort imgs name for readable"

    results = []
    for num, img_dir in tqdm(enumerate(imgs_list), total=len(imgs_list), desc='model runnning: '):
        image = imread(img_dir)
        image = cv2.resize(image, resize)
        image_tensor = img2tensor(image, device)
        with torch.no_grad():
            image_feat = model(image_tensor)
        results.append(image_feat.detach().cpu())

    remove_list = []
    for i in tqdm(range(len(results)),desc=f'checking: '):
        sim = []
        for j in range(len(results)):
            sim_item = torch.nn.functional.mse_loss(results[i], results[j])
            sim.append(sim_item)

        for num, score in enumerate(sim):
            if i in remove_list:
                continue

            if score < thred:
                if i != num:
                    if show_same:
                        print(f'\t\t {i + 1}.jpg == {num + 1}.jpg\tsimilarity score = {round(float(score), 2)}', )
                    remove_list.append(num)
    remove_list = list(set(remove_list))

    for index in sorted(remove_list, reverse=True):
        imgs_list.pop(index)

    save_list = [os.path.join(save_file, str(i + 1) + '.jpg') for i in range(len(imgs_list))]

    for i in range(len(imgs_list)):
        shutil.copy(imgs_list[i], save_list[i])
    print(f'Finish! Images are saved in {save_file}')
