import numpy as np
import torch.nn.functional as F

from .config import Config
from .models.models import LightTrackM_Subnet
from .utils import get_subwindow_tracking, python2round
from ..tracker import Tracker


class lighttrack(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super(lighttrack, self).__init__(checkpoint_path, use_cuda, 'lighttrack')

        self.cfg = Config()
        self.network = LightTrackM_Subnet(path_name=r'back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                                          stride=self.cfg.total_stride).to(self.device).eval()
        self.load_checkpoint()

    def init(self, image: np.array, bbox: list) -> None:
        self.im_h = image.shape[0]
        self.im_w = image.shape[1]

        cx, cy, w, h = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2), bbox[2], bbox[3]
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        self.grids(self.cfg)  # self.grid_to_search_x, self.grid_to_search_y

        wc_z = target_sz[0] + self.cfg.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self.cfg.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        self.avg_chans = np.mean(image, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(image, target_pos, self.cfg.exemplar_size, s_z, self.avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        self.network.template(z.cuda())

        self.window = np.outer(np.hanning(self.cfg.score_size), np.hanning(self.cfg.score_size))  # [17,17]

        self.target_pos = target_pos
        self.target_sz = target_sz

    def update(self, x_crops, target_pos, target_sz, scale_z):
        cls_score, bbox_pred = self.network.track(x_crops)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - self.cfg.window_influence) + self.window * self.cfg.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - self.cfg.instance_size // 2
        diff_ys = pred_ys - self.cfg.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.cfg.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, image: np.array) -> list:
        hc_z = self.target_sz[1] + self.cfg.context_amount * sum(self.target_sz)
        wc_z = self.target_sz[0] + self.cfg.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.cfg.exemplar_size / s_z
        d_search = (self.cfg.instance_size - self.cfg.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(image, self.target_pos, self.cfg.instance_size, python2round(s_x),
                                           self.avg_chans)
        self.x_crop = x_crop.clone()  # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)

        target_pos, target_sz, _ = self.update(x_crop.cuda(),
                                               self.target_pos,
                                               self.target_sz * scale_z,
                                               scale_z,
                                               )

        target_pos[0] = max(0, min(self.im_w, target_pos[0]))
        target_pos[1] = max(0, min(self.im_h, target_pos[1]))
        target_sz[0] = max(10, min(self.im_w, target_sz[0]))
        target_sz[1] = max(10, min(self.im_h, target_sz[1]))
        self.target_pos = target_pos
        self.target_sz = target_sz
        return [int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2), int(target_sz[0]),
                int(target_sz[1])]

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        # print('ATTENTION',p.instance_size,p.score_size)
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def normalize(self, x):
        """ input is in (C,H,W) format"""
        x /= 255
        x -= self.cfg.mean
        x /= self.cfg.std
        return x
