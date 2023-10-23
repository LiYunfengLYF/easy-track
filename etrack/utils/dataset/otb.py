import os
from ..load import txtread,seqread


class otbDataset(object):
    sequence_list = ['Basketball', 'Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3',
                     'BlurCar4', 'BlurFace', 'BlurOwl', 'Board', 'Bolt', 'Bolt2', 'Box', 'Boy', 'Car1', 'Car2',
                     'Car24', 'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Coke', 'Couple', 'Coupon', 'Crossing',
                     'Crowds', 'Dancer', 'Dancer2', 'David', 'David2', 'David3', 'Deer', 'Diving', 'Dog', 'Dog1',
                     'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football',
                     'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2', 'Human3',
                     'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman', 'Jogging_1', 'Jogging_2',
                     'Jump', 'Jumping', 'KiteSurf', 'Lemming', 'Liquor', 'Man', 'Matrix', 'Mhyang', 'MotorRolling',
                     'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Singer2', 'Skater',
                     'Skater2', 'Skating1', 'Skating2_1', 'Skating2_2', 'Skiing', 'Soccer', 'Subway', 'Surfer', 'Suv',
                     'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trans', 'Trellis', 'Twinnings', 'Vase', 'Walking',
                     'Walking2', 'Woman']

    def __init__(self, path):
        super(otbDataset, self).__init__()
        self.path = path
        self.seqs_info = [self.build_seq_info(i) for i in self.sequence_list]

    def build_seq_info(self, seq_name):
        imgs_dir = os.path.join(self.path, seq_name, 'img')
        gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.txt')

        if seq_name == 'Human4':
            gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.2.txt')
        if seq_name == 'Skating2_1':
            seq_name = 'Skating2'
            imgs_dir = os.path.join(self.path, seq_name, 'img')
            gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.1.txt')
        if seq_name == 'Skating2_2':
            seq_name = 'Skating2'
            imgs_dir = os.path.join(self.path, seq_name, 'img')
            gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.2.txt')
        if seq_name == 'Jogging_1':
            seq_name = 'Jogging'
            imgs_dir = os.path.join(self.path, seq_name, 'img')
            gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.1.txt')
        if seq_name == 'Jogging_2':
            seq_name = 'Jogging'
            imgs_dir = os.path.join(self.path, seq_name, 'img')
            gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.2.txt')

        return {'name': seq_name, 'imgs_dir': imgs_dir, 'gt_dir': gt_dir}

    def __getitem__(self, item):
        seq_info = self.seqs_info[item]
        imgs = seqread(seq_info['imgs_dir'])
        gt_txt = txtread(seq_info['gt_dir'], delimiter=[',', '\t'])

        return seq_info['name'], imgs, gt_txt

    def __len__(self):
        return len(self.sequence_list)
