import os
from tqdm import tqdm


def remove_timetxt(results_path):
    txt_list = os.listdir(results_path)
    remove_list = []

    for item in txt_list:
        if item.split('.')[-2].split('_')[-1] == 'time':
            remove_list.append(item)

    for i in tqdm(range(len(remove_list)), total=len(remove_list), desc='removing the *_time.txt: ', position=1):
        os.remove(os.path.join(results_path, remove_list[i]))

    print(f'Finish remove *_time.txt in {results_path}')
