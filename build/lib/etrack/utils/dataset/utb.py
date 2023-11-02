import os
from ..load import txtread,seqread


class utb180Dataset(object):
    sequence_list = ['Video_0001', 'Video_0002', 'Video_0003', 'Video_0004', 'Video_0005', 'Video_0006', 'Video_0007',
                     'Video_0008', 'Video_0009', 'Video_0010', 'Video_0011', 'Video_0012', 'Video_0013', 'Video_0014',
                     'Video_0015', 'Video_0016', 'Video_0017', 'Video_0018', 'Video_0019', 'Video_0020', 'Video_0021',
                     'Video_0022', 'Video_0023', 'Video_0024', 'Video_0025', 'Video_0026', 'Video_0027', 'Video_0028',
                     'Video_0029', 'Video_0030', 'Video_0031', 'Video_0032', 'Video_0033', 'Video_0034', 'Video_0035',
                     'Video_0036', 'Video_0037', 'Video_0038', 'Video_0039', 'Video_0040', 'Video_0041', 'Video_0042',
                     'Video_0043', 'Video_0044', 'Video_0045', 'Video_0046', 'Video_0047', 'Video_0048', 'Video_0049',
                     'Video_0050', 'Video_0051', 'Video_0052', 'Video_0053', 'Video_0054', 'Video_0055', 'Video_0056',
                     'Video_0057', 'Video_0058', 'Video_0059', 'Video_0060', 'Video_0061', 'Video_0062', 'Video_0063',
                     'Video_0064', 'Video_0065', 'Video_0066', 'Video_0067', 'Video_0068', 'Video_0069', 'Video_0070',
                     'Video_0071', 'Video_0072', 'Video_0073', 'Video_0074', 'Video_0075', 'Video_0076', 'Video_0077',
                     'Video_0078', 'Video_0079', 'Video_0080', 'Video_0081', 'Video_0082', 'Video_0083', 'Video_0084',
                     'Video_0085', 'Video_0086', 'Video_0087', 'Video_0088', 'Video_0089', 'Video_0090', 'Video_0091',
                     'Video_0092', 'Video_0093', 'Video_0094', 'Video_0095', 'Video_0096', 'Video_0097', 'Video_0098',
                     'Video_0099', 'Video_0100', 'Video_0101', 'Video_0102', 'Video_0103', 'Video_0104', 'Video_0105',
                     'Video_0106', 'Video_0107', 'Video_0108', 'Video_0109', 'Video_0110', 'Video_0111', 'Video_0112',
                     'Video_0113', 'Video_0114', 'Video_0115', 'Video_0116', 'Video_0117', 'Video_0118', 'Video_0119',
                     'Video_0120', 'Video_0121', 'Video_0122', 'Video_0123', 'Video_0124', 'Video_0125', 'Video_0126',
                     'Video_0127', 'Video_0128', 'Video_0129', 'Video_0130', 'Video_0131', 'Video_0132', 'Video_0133',
                     'Video_0134', 'Video_0135', 'Video_0136', 'Video_0137', 'Video_0138', 'Video_0139', 'Video_0140',
                     'Video_0141', 'Video_0142', 'Video_0143', 'Video_0144', 'Video_0145', 'Video_0151', 'Video_0152',
                     'Video_0153', 'Video_0154', 'Video_0155', 'Video_0156', 'Video_0157', 'Video_0158', 'Video_0159',
                     'Video_0160', 'Video_0161', 'Video_0162', 'Video_0163', 'Video_0164', 'Video_0165', 'Video_01',
                     'Video_02', 'Video_03', 'Video_04', 'Video_05', 'Video_06', 'Video_07', 'Video_08', 'Video_09',
                     'Video_10', 'Video_11', 'Video_12', 'Video_13', 'Video_14', 'Video_15', 'Video_16', 'Video_17',
                     'Video_18', 'Video_19', 'Video_20']

    def __init__(self, path):
        super(utb180Dataset, self).__init__()
        self.path = path
        self.seqs_info = [self.build_seq_info(i) for i in self.sequence_list]

    def build_seq_info(self, seq_name):
        imgs_dir = os.path.join(self.path, seq_name, 'imgs')
        gt_dir = os.path.join(self.path, seq_name, 'groundtruth_rect.txt')

        return {'name': seq_name, 'imgs_dir': imgs_dir, 'gt_dir': gt_dir}

    def __getitem__(self, item):
        seq_info = self.seqs_info[item]
        imgs = seqread(seq_info['imgs_dir'])
        gt_txt = txtread(seq_info['gt_dir'], delimiter=[',', '\t'])

        return seq_info['name'], imgs, gt_txt

    def __len__(self):
        return len(self.sequence_list)
