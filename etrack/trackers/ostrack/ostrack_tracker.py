import torch
import numpy as np
from ..tracker import Tracker
from .config import cfg_256, cfg_384
from .models import OSTrack256, OSTrack384
from .utils import Preprocessor, hann2d, sample_target, clip_box, generate_mask_cond, transform_image_to_crop

__all__ = ['ostrack256', 'ostrack384']


class ostrack256(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'ostrack256')

        self.cfg = cfg_256()
        self.network = OSTrack256().to(self.device).eval()
        self.load_checkpoint()

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.search_size // self.cfg.stride

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

    def init(self, image: np.array, bbox: list) -> None:
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, bbox, self.cfg.template_factor,
                                                                output_sz=self.cfg.template_size)
        template = self.preprocessor.process(z_patch_arr)

        self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.ce_loc:
            template_bbox = self.transform_bbox_to_crop(bbox, resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save states
        self.state = bbox

    def track(self, image: np.array) -> list:
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.cfg.search_factor,
                                                                output_sz=self.cfg.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(
                template=self.z_dict1, search=x_dict, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.cfg.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        return self.state

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        # box_in: list [x1, y1, w, h], not normalized
        # box_extract: same as box_in
        # out bbox: Torch.tensor [1, 1, 4], x1y1wh, normalized
        if crop_type == 'template':
            crop_sz = torch.Tensor([self.cfg.template_size, self.cfg.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.cfg.search_size, self.cfg.search_size])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox


class ostrack384(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'ostrack384')

        self.cfg = cfg_384()
        self.network = OSTrack384().to(self.device).eval()
        self.load_checkpoint()

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.search_size // self.cfg.stride

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

    def init(self, image, bbox):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, bbox, self.cfg.template_factor,
                                                                output_sz=self.cfg.template_size)
        template = self.preprocessor.process(z_patch_arr)

        self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.ce_loc:
            template_bbox = self.transform_bbox_to_crop(bbox, resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save states
        self.state = bbox

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.cfg.search_factor,
                                                                output_sz=self.cfg.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(
                template=self.z_dict1, search=x_dict, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.cfg.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        return self.state

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        # box_in: list [x1, y1, w, h], not normalized
        # box_extract: same as box_in
        # out bbox: Torch.tensor [1, 1, 4], x1y1wh, normalized
        if crop_type == 'template':
            crop_sz = torch.Tensor([self.cfg.template_size, self.cfg.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.cfg.search_size, self.cfg.search_size])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox
