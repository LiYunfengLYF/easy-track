from abc import abstractmethod
from collections import deque
from typing import Dict, Any, Tuple, Callable

import albumentations as albu
import numpy as np
import torch

from .box_coder import FEARBoxCoder, TrackerDecodeResult
from .config import cfg
from .models import FEARModel
from .utils import TrackingState, to_device, clamp_bbox, get_extended_crop
from ..tracker import Tracker


class fearxxs(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True) -> None:
        super().__init__(checkpoint_path, use_cuda, 'fearxs')
        self.tracker = fearxxs_tracker(checkpoint_path, use_cuda)

    def init(self, image: np.array, bbox: list):
        self.tracker.init(image, bbox)

    def track(self, image: np.array):
        return self.tracker.update(image)


class fearxxs_tracker(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True) -> None:
        super().__init__(checkpoint_path, use_cuda, 'fearxs')

        self.cuda_id = 0
        self.tracking_config = cfg()
        self.tracking_state = TrackingState()
        self.network = FEARModel().to(self.device).eval()
        self.load_checkpoint()

        self.box_coder = self.get_box_coder(self.tracking_config, self.cuda_id)
        self._template_features = None
        self._template_transform = self._get_default_transform(img_size=self.tracking_config.template_size)
        self._search_transform = self._get_default_transform(img_size=self.tracking_config.instance_size)
        self.window = self._get_tracking_window(self.tracking_config.windowing, self.tracking_config.score_size)
        self.to_device(self.cuda_id)

    def init(self, image: np.ndarray, bbox: np.array, **kwargs) -> None:
        """
        args:
            img(np.ndarray): RGB image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        rect = clamp_bbox(bbox, image.shape)
        self.tracking_state.bbox = rect
        self.tracking_state.paths = deque([rect], maxlen=10)
        self.tracking_state.mean_color = np.mean(image, axis=(0, 1))
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            offset=self.tracking_config.template_bbox_offset,
            crop_size=self.tracking_config.template_size,
        )
        self._template_features = self.get_template_features(image, rect)

    def track(self, search_crop: np.ndarray):
        search_crop = self._preprocess_image(search_crop, self._search_transform)
        track_result = self.network.model.track(search_crop, self._template_features)
        return self._postprocess(track_result=track_result)

    def get_template_features(self, image, rect):
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            offset=self.tracking_config.template_bbox_offset,
            crop_size=self.tracking_config.template_size,
        )
        img = self._preprocess_image(template_crop, self._template_transform)
        return self.network.model.get_features(img)

    def update(self, image: np.ndarray):
        """
        args:
            img(np.ndarray): RGB image
        return:
            bbox(np.array):[x, y, width, height]
        """
        search_crop, search_bbox, padded_bbox = get_extended_crop(
            image=image,
            bbox=self.tracking_state.bbox,
            crop_size=self.tracking_config.instance_size,
            offset=self.tracking_config.search_context,
            padding_value=self.tracking_state.mean_color,
        )
        self.tracking_state.mapping = padded_bbox
        self.tracking_state.prev_size = search_bbox[2:]
        pred_bbox, _ = self.track(search_crop)
        pred_bbox = self._rescale_bbox(pred_bbox, self.tracking_state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, image.shape)
        self.tracking_state.bbox = pred_bbox
        self.tracking_state.paths.append(pred_bbox)
        return pred_bbox

    @staticmethod
    def _array_to_batch(x: np.ndarray) -> torch.Tensor:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    @abstractmethod
    def get_box_coder(self, tracking_config, cuda_id: int = 0):
        pass

    def to_device(self, cuda_id):
        self.cuda_id = cuda_id
        self.window = to_device(self.window, cuda_id)
        self.box_coder = self.box_coder.to_device(self.cuda_id)

    @staticmethod
    def _get_tracking_window(windowing: str, score_size: int) -> torch.Tensor:
        """

        :param windowing: str - window creation type
        :param score_size: int - size of classification map
        :return: window: np.array
        """
        if windowing == "cosine":
            return torch.from_numpy(np.outer(np.hanning(score_size), np.hanning(score_size)))
        return torch.ones(int(score_size), int(score_size))

    @staticmethod
    def _get_default_transform(img_size):
        pipeline = albu.Compose(
            [
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def process(a):
            r = pipeline(image=a)
            return r["image"]

        return process

    def _rescale_bbox(self, bbox: np.array, padded_box) -> np.array:
        w_scale = padded_box[2] / self.tracking_config.instance_size
        h_scale = padded_box[3] / self.tracking_config.instance_size
        bbox[0] = round(bbox[0] * w_scale + padded_box[0])
        bbox[1] = round(bbox[1] * h_scale + padded_box[1])
        bbox[2] = max(3, round(bbox[2] * w_scale))
        bbox[3] = max(3, round(bbox[3] * h_scale))
        return list(map(int, bbox))

    def _get_scale(self, bbox: np.ndarray) -> int:
        wc_z = bbox[2] + self.tracking_config.search_context * sum(bbox[2:])
        hc_z = bbox[3] + self.tracking_config.search_context * sum(bbox[2:])
        return max(round(np.sqrt(wc_z * hc_z)), 1)

    def _preprocess_image(self, image: np.ndarray, transform: Callable) -> torch.Tensor:
        img = transform(image[:, :, :3])
        if image.shape[2] > 3:
            img = np.concatenate([img, image[:, :, 3:]], axis=2)
        img = self._array_to_batch(img).float()
        img = to_device(img, cuda_id=self.cuda_id)
        return img

    def reset(self) -> None:
        self._template_features = None

    def _smooth_size(self, size: np.array, prev_size: np.array, lr: float) -> Tuple[float, float]:
        """
        Tracking smoothing logic matches the code of Siamese Tracking
        https://www.robots.ox.ac.uk/~luca/siamese-fc.html
        :param size: np.array([w, h]) - predicted bbox size
        :param prev_size: np.array([w, h]) - bbox size on previous frame
        :param lr: float - smoothing learning rate
        :return: Tuple[float, float] - smoothed size
        """
        size = size * lr
        prev_size = prev_size * (1 - lr)
        w = prev_size[0] + lr * (size[0] + prev_size[0])
        h = prev_size[1] + lr * (size[1] + prev_size[1])
        return w, h

    def _get_point_offset(self, pred_bbox: np.array) -> Tuple[float, float]:
        pred_xs = pred_bbox[0] + (pred_bbox[2] / 2)
        pred_ys = pred_bbox[1] + (pred_bbox[3] / 2)

        diff_xs = pred_xs - self.tracking_config.instance_size // 2
        diff_ys = pred_ys - self.tracking_config.instance_size // 2
        return diff_xs, diff_ys

    def _postprocess_bbox(
            self, decoded_info: TrackerDecodeResult, cls_score: np.array, penalty: Any = None
    ) -> np.array:
        pred_bbox = np.squeeze(decoded_info.bbox.cpu().numpy())
        if True:
            return pred_bbox

    def _confidence_postprocess(
            self, cls_score: np.ndarray, regression_map: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """

        :param cls_score: torch.Tensor - classification score
        :param pred_location: torch.Tensor - predicted locations
        :param prev_size: np.array - size from previous frame
        :return: penalty_score: np.ndarray - updated cls_score
        """
        if True:
            return cls_score, None

    def get_box_coder(self, tracking_config, cuda_id: int = 0):
        return FEARBoxCoder(tracker_config=tracking_config)

    def _postprocess(self, track_result: Dict[str, torch.Tensor]) -> Tuple[np.array, float]:
        cls_score = track_result['TARGET_CLASSIFICATION_KEY'].detach().float().sigmoid()
        regression_map = track_result['TARGET_REGRESSION_LABEL_KEY'].detach().float()
        classification_map, penalty = self._confidence_postprocess(cls_score=cls_score, regression_map=regression_map)
        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=classification_map,
            regression_map=track_result['TARGET_REGRESSION_LABEL_KEY'],
            use_sigmoid=False,
        )
        cls_score = np.squeeze(cls_score)
        pred_bbox = self._postprocess_bbox(decoded_info=decoded_info, cls_score=cls_score, penalty=penalty)
        r_max, c_max = decoded_info.pred_coords[0]
        return pred_bbox, cls_score[r_max, c_max]
