import etrack
from etrack.test.testboard import info_board

checkpoint_path = r'/home/liyunfeng/code/dev/easy-track/dev/vitb_384_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar'

etrack.extract_weights_from_checkpoint(checkpoint_file=checkpoint_path,name='ostrack384')
from etrack import lightfc
tracker = lightfc()