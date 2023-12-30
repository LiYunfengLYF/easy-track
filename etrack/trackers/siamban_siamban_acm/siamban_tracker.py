import cv2
import numpy as np
import torch
from ..tracker import Tracker
from .config import cfg_siamban, cfg_siamban_acm
from .models import SiamBAN_Resnet, SiamBAN_ACM_Resnet
from .utils import corner2center


class _siambantracker(Tracker):

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.to(self.device)
        return im_patch


class siamban(_siambantracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'siamban')

        self.cfg = cfg_siamban()
        self.network = SiamBAN_Resnet().eval().to(self.device)
        self.load_checkpoint()

        self.cls_out_channels = self.cfg.ban_cls_out_channels
        self.score_size = (self.cfg.track_instance_size - self.cfg.track_exemplar_size) // \
                          self.cfg.point_stride + 1 + self.cfg.track_base_size
        self.window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size)).flatten()
        self.points = self.generate_points(self.cfg.point_stride, self.score_size)

    def init(self, img: np.array, bbox: list) -> None:
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + self.cfg.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.cfg.track_context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, self.cfg.track_exemplar_size, s_z, self.channel_average)
        self.network.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + self.cfg.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.cfg.track_context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.track_exemplar_size / s_z
        s_x = s_z * (self.cfg.track_instance_size / self.cfg.track_exemplar_size)
        x_crop = self.get_subwindow(img, self.center_pos, self.cfg.track_instance_size, round(s_x),
                                    self.channel_average)

        outputs = self.network.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.track_penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.track_window_influence) + \
                 self.window * self.cfg.track_window_influence
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.track_lr

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        return bbox


class siamban_acm(_siambantracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'siamban_acm')

        self.cfg = cfg_siamban_acm()
        self.network = SiamBAN_ACM_Resnet().eval().to(self.device)
        self.load_checkpoint()

        self.init_window(self.cfg.track_instance_size)

    def init(self, img: np.array, bbox: list) -> None:
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + self.cfg.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.cfg.track_context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, self.cfg.track_exemplar_size, s_z, self.channel_average)

        r = self.cfg.track_exemplar_size / s_z
        t_bbox = np.array(bbox)
        t_bbox = t_bbox * r
        t_bbox[0] = self.cfg.track_exemplar_size / 2
        t_bbox[1] = self.cfg.track_exemplar_size / 2

        t_bbox = torch.from_numpy(t_bbox).unsqueeze(0).type(torch.float32).cuda()

        self.network.template(z_crop, t_bbox)
        self.fast_motion = False

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + self.cfg.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.cfg.track_context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.track_exemplar_size / s_z

        instance_size = self.cfg.track_instance_size
        if self.fast_motion:
            instance_size = self.cfg.track_instance_size * 2

        self.init_window(instance_size)

        s_x = s_z * (instance_size / self.cfg.track_exemplar_size)

        x_crop = self.get_subwindow(img, self.center_pos, instance_size, round(s_x), self.channel_average)

        outputs = self.network.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.track_penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.track_window_influence) + \
                 self.window * self.cfg.track_window_influence
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.track_lr

        if bbox[0] > s_z / 2 or bbox[1] > s_z / 2:
            self.fast_motion = True

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        return bbox

    def init_window(self, instance_size):
        self.score_size = (instance_size - self.cfg.track_exemplar_size) // \
                          self.cfg.point_stride + 1 + self.cfg.track_base_size
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = self.cfg.ban_cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(self.cfg.point_stride, self.score_size)
