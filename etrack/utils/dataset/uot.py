import os
from ..load import txtread, seqread


class uot100Dataset(object):
    sequence_list = ['AntiguaTurtle', 'ArmyDiver1', 'ArmyDiver2', 'ArmyDiver3', 'Ballena', 'BallisticMissile1',
                     'BallisticMissile2', 'BlueFish1', 'BlueFish2', 'BoySwimming', 'CenoteAngelita', 'CleverOctopus',
                     'ClickerAndTarget', 'CoconutOctopus1', 'CoconutOctopus2', 'ColourChangingSquid', 'CoralGardenSea1',
                     'CoralGardenSea2', 'CrabTrap', 'CrayFish', 'CressiGuillaumeNeri1', 'CressiGuillaumeNeri2',
                     'Cuttlefish', 'DeepOceanLostWorld', 'DeepSeaFish', 'DefenseInTheSea1', 'DefenseInTheSea2',
                     'Diving360Degree1', 'Diving360Degree2', 'Diving360Degree3', 'Dolphin1', 'Dolphin2',
                     'ElephantSeals', 'FightingEels1', 'FightingEels2', 'FightToDeath', 'Fisherman', 'FishFollowing',
                     'FishingAdventure', 'FishingBait', 'FlukeFishing1', 'FlukeFishing2', 'FreeDiver1', 'FreeDiver2',
                     'GarryFish', 'GiantCuttlefish1', 'GiantCuttlefish2', 'GreenMoreyEel1', 'GreenMoreyEel2',
                     'GreenMoreyEel3', 'GuillaumeNery', 'HappyTurtle1', 'HappyTurtle2', 'HappyTurtle3', 'HeartShape',
                     'HoverFish1', 'HoverFish2', 'JerkbaitBites', 'Kleptopus1', 'Kleptopus2', 'LargemouthBass',
                     'LittleMonster', 'Lobsters1', 'Lobsters2', 'MantaRescue1', 'MantaRescue2', 'MantaRescue3',
                     'MantaRescue4', 'MantisShrimp', 'MississippiFish', 'MonsterCreature1', 'MonsterCreature2',
                     'MuckySecrets1', 'MuckySecrets2', 'MythBusters', 'NeryClimbing', 'OceanFloorSensor', 'Octopus1',
                     'Octopus2', 'PinkFish', 'PlayingTurtle', 'RedSeaReptile', 'Rocketman', 'ScubaDiving1',
                     'ScubaDiving2', 'SeaDiver', 'SeaDragon', 'SeaTurtle1', 'SeaTurtle2', 'SeaTurtle3',
                     'SharkCloseCall1', 'SharkCloseCall2', 'SharkSuckers1', 'SharkSuckers2', 'Skagerrak', 'SofiaRocks1',
                     'SofiaRocks2', 'Steinlager', 'Submarine', 'ThePassage', 'WallEye', 'WhaleAtBeach1',
                     'WhaleAtBeach2', 'WhaleDiving', 'WhiteShark', 'WolfTrolling']

    def __init__(self, path):
        super(uot100Dataset, self).__init__()
        self.path = path
        self.seqs_info = [self.build_seq_info(i) for i in self.sequence_list]

    def build_seq_info(self, seq_name):
        imgs_dir = os.path.join(self.path, seq_name, 'img')
        gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.txt')

        return {'name': seq_name, 'imgs_dir': imgs_dir, 'gt_dir': gt_dir}

    def __getitem__(self, item):
        seq_info = self.seqs_info[item]
        imgs = seqread(seq_info['imgs_dir'])
        gt_txt = txtread(seq_info['gt_dir'], delimiter=[',', '\t'])

        return seq_info['name'], imgs, gt_txt

    def __len__(self):
        return len(self.sequence_list)
