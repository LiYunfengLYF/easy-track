def tqdm_update(bar, seq_id, length, seq_name):
    bar.set_description(f'[{seq_id + 1}/{length}] {seq_name} ')
    bar.update(1)
