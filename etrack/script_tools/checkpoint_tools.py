import os
import torch
def extract_weights_from_checkpoint(checkpoint_file, out_file=None, name=None, key=None):
    """
    Description
        extract model's weight from checkpoint
        if out_file is None, save weights to current file
        if name is None, use current checkpoint name
        if key is None, use default names: 'net', 'network', 'model'

    Params:
        checkpoint_file:    checkpoint file
        out_file:           save weight file
        name:               save name, str type
        key:                key name, str type

    """
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
    """
    Description
        show checkpoint keys

    Params:
        checkpoint_file:    checkpoint file

    """


    try:
        state_dict = torch.load(checkpoint_file)
    except Exception as e:
        print(e)
        return

    print('Keys:')
    for key in state_dict:
        print('\t',key)

