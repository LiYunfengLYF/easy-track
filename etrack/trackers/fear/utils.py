from typing import Tuple, Any, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch


@torch.no_grad()
def make_grid(score_size: int, total_stride: int, instance_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each element of feature map on input search image
    :return: H*W*2 (position for each element)
    """

    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )

    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    grid_x = torch.from_numpy(grid_x[np.newaxis, :, :])
    grid_y = torch.from_numpy(grid_y[np.newaxis, :, :])
    return grid_x, grid_y


def unravel_index(index: Any, shape: Tuple[int, int]) -> Tuple[int, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class TrackingState:
    def __init__(self) -> None:
        super().__init__()
        self.frame_h = 0
        self.frame_w = 0
        self.bbox: Optional[np.array] = None
        self.mapping: Optional[np.array] = None
        self.prev_size = None
        self.mean_color = None

    def save_frame_shape(self, frame: np.ndarray) -> None:
        self.frame_h = frame.shape[0]
        self.frame_w = frame.shape[1]


def to_device(x: Union[torch.Tensor, torch.nn.Module], cuda_id: int = 0) -> torch.Tensor:
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def ensure_bbox_boundaries(bbox: np.array, img_shape: Tuple[int, int]) -> np.array:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def clamp_bbox(bbox: np.array, shape: Tuple[int, int], min_side: int = 3) -> np.array:
    bbox = ensure_bbox_boundaries(bbox, img_shape=shape)
    x, y, w, h = bbox
    img_h, img_w = shape[0], shape[1]
    if w < min_side:
        w = min_side
        x -= max(0, x + w - img_w)
    if h < min_side:
        h = min_side
        y -= max(0, y + h - img_h)
    return np.array([x, y, w, h])


def get_extended_crop(
        image: np.array, bbox: np.array, crop_size: int, offset: float, padding_value: np.array = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Extend bounding box by {offset} percentages, pad all sides, rescale to fit {crop_size} and pad all sides to make
    {side}x{side} crop
    Args:
        image: np.array
        bbox: np.array - in xywh format
        crop_size: int - output crop size
        offset: float - how much percentages bbox extend
        padding_value: np.array - value to pad

    Returns:
        crop_image: np.array
        crop_bbox: np.array
    """
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))
    bbox_params = {"format": "coco", "min_visibility": 0, "label_fields": ["category_id"], "min_area": 0}
    resize_aug = A.Compose([A.Resize(crop_size, crop_size)], bbox_params=bbox_params)
    context = extend_bbox(bbox, offset)
    pad_left, pad_top = max(-context[0], 0), max(-context[1], 0)
    pad_right, pad_bottom = max(context[0] + context[2] - image.shape[1], 0), max(
        context[1] + context[3] - image.shape[0], 0
    )
    crop = image[
           context[1] + pad_top: context[1] + context[3] - pad_bottom,
           context[0] + pad_left: context[0] + context[2] - pad_right,
           ]

    padded_crop = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padding_value
    )
    padded_bbox = np.array([bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]])
    padded_bbox = ensure_bbox_boundaries(padded_bbox, img_shape=padded_crop.shape[:2])
    result = resize_aug(image=padded_crop, bboxes=[padded_bbox], category_id=["bbox"])
    image, bbox = result["image"], np.array(result["bboxes"][0])
    return image, bbox, context


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def limit(radius: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    if isinstance(radius, torch.Tensor):
        return torch.maximum(radius, 1.0 / radius)
    return np.maximum(radius, 1.0 / radius)


def squared_size(w: int, h: int) -> Union[torch.Tensor, float]:
    pad = (w + h) * 0.5
    size = (w + pad) * (h + pad)
    if isinstance(size, torch.Tensor):
        return torch.sqrt(size)
    return np.sqrt(size)
