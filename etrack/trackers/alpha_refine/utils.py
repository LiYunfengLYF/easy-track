import cv2
import math
import numpy as np
import torch


def sample_target_SE(im, target_bb, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    x, y, w, h = target_bb

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - ws * 0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape) == 2:
            im_crop_padded_rsz = im_crop_padded_rsz[..., np.newaxis]
        return im_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, 1.0, 1.0


def map_mask_back(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    H, W = (im.shape[0], im.shape[1])
    base = np.zeros((H, W))
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - ws * 0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    '''pad base'''
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    '''Resize mask'''
    mask_rsz = cv2.resize(mask, (ws, hs))
    '''fill region with mask'''
    base_padded[y1 + y1_pad:y2 + y1_pad, x1 + x1_pad:x2 + x1_pad] = mask_rsz.copy()
    '''crop base_padded to get final mask'''
    final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
    assert (final_mask.shape == (H, W))
    return final_mask


'''Added on 2019.12.23'''


def transform_image_to_crop_SE(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor_h: float,
                               resize_factor_w: float,
                               crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_xc = (crop_sz[0] - 1) / 2 + (box_in_center[0] - box_extract_center[0]) * resize_factor_w
    box_out_yc = (crop_sz[0] - 1) / 2 + (box_in_center[1] - box_extract_center[1]) * resize_factor_h
    box_out_w = box_in[2] * resize_factor_w
    box_out_h = box_in[3] * resize_factor_h

    '''2019.12.28 为了避免出现(x1,y1)小于0,或者(x2,y2)大于256的情况,这里我对它们加上了一些限制'''
    max_sz = crop_sz[0].item()
    box_out_x1 = torch.clamp(box_out_xc - 0.5 * box_out_w, 0, max_sz)
    box_out_y1 = torch.clamp(box_out_yc - 0.5 * box_out_h, 0, max_sz)
    box_out_x2 = torch.clamp(box_out_xc + 0.5 * box_out_w, 0, max_sz)
    box_out_y2 = torch.clamp(box_out_yc + 0.5 * box_out_h, 0, max_sz)
    box_out_w_new = box_out_x2 - box_out_x1
    box_out_h_new = box_out_y2 - box_out_y1
    box_out = torch.stack((box_out_x1, box_out_y1, box_out_w_new, box_out_h_new))
    return box_out


def mask2bbox(mask, ori_bbox, MASK_THRESHOLD=0.5):
    target_mask = (mask > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boundingRect(polygon)
    else:  # empty mask
        prbox = ori_bbox
    return np.array(prbox).astype(np.float64)


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh
