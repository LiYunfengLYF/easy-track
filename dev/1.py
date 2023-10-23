from etrack import siamfc, run_sequence

tracker = siamfc()

run_sequence(tracker, seq_file=r'/media/liyunfeng/CV2/data/sot/otb/Basketball/img', select_roi=True)
