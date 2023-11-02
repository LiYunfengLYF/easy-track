import torch
import got10k.experiments
from etrack import starks50, run_sequence, quick_start, starkst50, starkst101, ostrack256
import etrack
tracker = starks50()
# seq_file = r'/home/liyunfeng/Downloads/SonarPolarizationLight/fusion/light/9_sonar10_mooringmine/Sensor2_Underwater_Camera_6000'
seq_file = r'/media/liyunfeng/CV2/data/sot/otb/Basketball/img'
seq_gt_file = r'/media/liyunfeng/CV2/data/sot/otb/Basketball/groundtruth_rect.txt'

run_sequence(tracker, seq_file=seq_file, gt_file=seq_gt_file, speed=100, visual=True, report_performance=True)


# etrack.report_seq_performance(seq_gt_file,seq_gt_file)