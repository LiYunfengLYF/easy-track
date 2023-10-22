import os
from ..load import txtread


class lasotextDataset(object):
    sequence_list = [
        'atv-1', 'atv-2', 'atv-3', 'atv-4', 'atv-5', 'atv-6', 'atv-7', 'atv-8', 'atv-9', 'atv-10', 'badminton-1',
        'badminton-2', 'badminton-3', 'badminton-4', 'badminton-5', 'badminton-6', 'badminton-7', 'badminton-8',
        'badminton-9', 'badminton-10', 'cosplay-1', 'cosplay-10', 'cosplay-2', 'cosplay-3', 'cosplay-4', 'cosplay-5',
        'cosplay-6', 'cosplay-7', 'cosplay-8', 'cosplay-9', 'dancingshoe-1', 'dancingshoe-2', 'dancingshoe-3',
        'dancingshoe-4', 'dancingshoe-5', 'dancingshoe-6', 'dancingshoe-7', 'dancingshoe-8', 'dancingshoe-9',
        'dancingshoe-10', 'footbag-1', 'footbag-2', 'footbag-3', 'footbag-4', 'footbag-5', 'footbag-6', 'footbag-7',
        'footbag-8', 'footbag-9', 'footbag-10', 'frisbee-1', 'frisbee-2', 'frisbee-3', 'frisbee-4', 'frisbee-5',
        'frisbee-6', 'frisbee-7', 'frisbee-8', 'frisbee-9', 'frisbee-10', 'jianzi-1', 'jianzi-2', 'jianzi-3',
        'jianzi-4', 'jianzi-5', 'jianzi-6', 'jianzi-7', 'jianzi-8', 'jianzi-9', 'jianzi-10', 'lantern-1', 'lantern-2',
        'lantern-3', 'lantern-4', 'lantern-5', 'lantern-6', 'lantern-7', 'lantern-8', 'lantern-9', 'lantern-10',
        'misc-1', 'misc-2', 'misc-3', 'misc-4', 'misc-5', 'misc-6', 'misc-7', 'misc-8', 'misc-9', 'misc-10',
        'opossum-1', 'opossum-2', 'opossum-3', 'opossum-4', 'opossum-5', 'opossum-6', 'opossum-7', 'opossum-8',
        'opossum-9', 'opossum-10', 'paddle-1', 'paddle-2', 'paddle-3', 'paddle-4', 'paddle-5', 'paddle-6', 'paddle-7',
        'paddle-8', 'paddle-9', 'paddle-10', 'raccoon-1', 'raccoon-2', 'raccoon-3', 'raccoon-4', 'raccoon-5',
        'raccoon-6', 'raccoon-7', 'raccoon-8', 'raccoon-9', 'raccoon-10', 'rhino-1', 'rhino-2', 'rhino-3', 'rhino-4',
        'rhino-5', 'rhino-6', 'rhino-7', 'rhino-8', 'rhino-9', 'rhino-10', 'skatingshoe-1', 'skatingshoe-2',
        'skatingshoe-3', 'skatingshoe-4', 'skatingshoe-5', 'skatingshoe-6', 'skatingshoe-7', 'skatingshoe-8',
        'skatingshoe-9', 'skatingshoe-10', 'wingsuit-1', 'wingsuit-2', 'wingsuit-3', 'wingsuit-4', 'wingsuit-5',
        'wingsuit-6', 'wingsuit-7', 'wingsuit-8', 'wingsuit-9', 'wingsuit-10']

    def __init__(self, path):
        super(lasotextDataset, self).__init__()
        self.path = path
        self.seqs_info = [self.build_seq_info(i) for i in self.sequence_list]

    def build_seq_info(self, seq_name):
        imgs_dir = os.path.join(self.path, seq_name.split('-')[0], seq_name, 'img')
        gt_dir = os.path.join(self.path, seq_name.split('-')[0], seq_name, 'groundtruth.txt')

        return {
            'name': seq_name,
            'imgs_dir': imgs_dir,
            'gt_dir': gt_dir
        }

    def __getitem__(self, item):
        seq_info = self.seqs_info[item]
        imgs = [os.path.join(seq_info['imgs_dir'], i) for i in (os.listdir(seq_info['imgs_dir']))]
        gt_txt = txtread(seq_info['gt_dir'], delimiter=[',', '\t'])

        return seq_info['name'], imgs, gt_txt

    def __len__(self):
        return len(self.sequence_list)
