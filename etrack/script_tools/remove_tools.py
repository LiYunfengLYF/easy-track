import os
import cv2
import torch
import shutil
import logging
from tqdm import tqdm
from .utils import mobilenet_v2
from ..utils import imread, seqread, img2tensor

def remove_timetxt(results_file):
    """
    Description
        Remove *_time.txt files from results_file

    Params:
        results_file:   file path

    """
    txt_list = os.listdir(results_file)
    remove_list = []

    for item in txt_list:
        if item.split('.')[-2].split('_')[-1] == 'time':
            remove_list.append(item)

    for i in tqdm(range(len(remove_list)), total=len(remove_list), desc='removing the *_time.txt: ', position=1):
        os.remove(os.path.join(results_file, remove_list[i]))

    print(f'Finish remove *_time.txt in {results_file}')


def remove_same_img(file, save_file, checkpoint_path=None, device='cuda:0', resize=(320, 640), thred=0.4,
                    show_same=False, start=1):
    """
    Description
        Remove same images in file and sort and save the rest images in save file
        It resizes input image to (320,640)(default) and uses MobileNetV2 to extract feature, then calc the similarity
        You need to sign the checkpoint_path of mobilenet_v2-b0353104.pth (from torchvision) and thred (default is 0.4)
        if not sign checkpoint path, it will search for weights in the etrack_checkpoints directory of the running .py file
        show_same=True will show the same image pair

    Params:
        results_file:       file path
        save_file:          file path
        checkpoint_path:    checkpoint path

    """
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
    for i in tqdm(range(len(results)), desc=f'checking: '):
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

    save_list = [os.path.join(save_file, str(i + start) + '.jpg') for i in range(len(imgs_list))]

    for i in range(len(imgs_list)):
        shutil.copy(imgs_list[i], save_list[i])
    print(f'Finish! Images are saved in {save_file}')
