import torch

from etrack import starks50, run_sequence,quick_start,starkst50,starkst101

tracker = starkst101()
seq_file =r'/home/liyunfeng/Downloads/SonarPolarizationLight/fusion/light/11_sonar15_mooringmine/Sensor1_OculusMD750d/image'
# tracker.network.load_state_dict(torch.load(r'/home/liyunfeng/code/project2/Stark/checkpoints/train/stark_s/baseline/STARKS_ep0500.pth.tar')['net'])
quick_start(tracker, seq_file=seq_file, speed=50)
#

from etrack import seqread