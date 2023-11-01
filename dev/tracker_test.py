import torch

from etrack import starks50, run_sequence, quick_start, starkst50, starkst101, ostrack256

tracker = starks50()
seq_file = r'/home/liyunfeng/Downloads/SonarPolarizationLight/fusion/light/9_sonar10_mooringmine/Sensor2_Underwater_Camera_6000'

run_sequence(tracker, seq_file=seq_file, speed=100, visual=True)
