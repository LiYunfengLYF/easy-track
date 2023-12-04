import os
from tqdm import tqdm


def trans_txt_delimiter(txt_file: str, out_file: str = None, delimiter: str = ',', new_delimiter: str = None,
                        with_time: bool = True) -> None:
    """
    Description
        transfer the delimiter of txt and save txt in out_file
        if out_file is None, out_file = txt_file
        if new_delimiter is None, new_delimiter = delimiter
        if with time is True, copy *_time.txt to out_file

    Params:
        txt_file:   source txt file
        out_file:   save txt file
        format:     ',' or '\t'
        new_format: ',' or '\t'
        with_time:  True or False

    """
    out_file = txt_file if out_file is None else out_file
    new_delimiter = format if new_delimiter is None else new_delimiter

    assert delimiter in ['\t', ',']
    assert new_delimiter in ['\t', ',']

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
                sline = line.replace(delimiter, new_delimiter)
                new_lines.append(sline)
        with open(save_results[idx], 'w') as f:
            f.writelines(new_lines)
