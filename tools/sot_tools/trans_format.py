import os
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


if __name__ == '__main__':
    trans_txt_format('a', 'n')
