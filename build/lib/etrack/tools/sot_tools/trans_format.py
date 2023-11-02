import os

import torch
from tqdm import tqdm


def trans_txt_format(txt_file, out_file=None, format=',', new_format=None, with_time=True):
    out_file = txt_file if out_file is None else out_file
    new_format = format if new_format is None else new_format

    assert format in ['\t', ',']
    assert new_format in ['\t', ',']

    if with_time:
        result_item = []
        for i in os.listdir(txt_file):

            if i.split('.')[-2].split('_')[-1] != 'time':
                result_item.append(i)
    else:
        result_item = os.listdir(txt_file)

    origin_results = [os.path.join(txt_file, item) for item in result_item]
    save_results = [os.path.join(out_file, item) for item in result_item]

    for idx, result in tqdm(enumerate(origin_results), total=len(origin_results), desc=f'processing :', position=0):
        new_lines = []

        with open(result, 'r') as f:
            lines = f.readlines()
            for line in lines:
                sline = line.replace(format, new_format)
                new_lines.append(sline)
        with open(save_results[idx], 'w') as f:
            f.writelines(new_lines)


def trans_checkpoint_keys(checkpoint_file, out_file=None, name=None, key=None):
    out_file = os.getcwd() if out_file is None else out_file
    state_dict = []

    if name is None:
        name_list = checkpoint_file.split('/')
        if len(name_list) == 0:
            name_list = checkpoint_file.split('\\')
            if len(name_list) == 0:
                raise print('Illegal checkpoint path')
        name = name_list[-1]
    else:
        if len(name.split('.')) == 1:
            name += '.pth'
        else:
            if not name.split('.') in ['pth', 'pth.tar', ]:
                raise print(f'Only support *.pth, *.pth.tar format')
    try:
        if key is None:
            total_state_dict = torch.load(checkpoint_file)
            for key_item in ['net', 'network', 'model']:
                state_dict = total_state_dict[key_item]
                print(f'\tloading state_dict from key: {key_item}')
                break
        else:
            state_dict = torch.load(checkpoint_file)[key]

        if state_dict is []:
            print('state_dict is empty!!!')
            raise NotImplementedError

    except Exception as e:
        print(e)
        print(f'{state_dict}')
        return

    save_path = os.path.join(out_file, name)
    torch.save(state_dict, save_path)
    print(f'\t{name} save to {save_path}')


def show_checkpoint_keys(checkpoint_file):
    try:
        state_dict = torch.load(checkpoint_file)
    except Exception as e:
        print(e)
        return

    print('Keys:')
    for key in state_dict:
        print('\t',key)


if __name__ == '__main__':
    trans_txt_format('a', 'n')
