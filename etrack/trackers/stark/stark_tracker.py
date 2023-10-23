from copy import deepcopy

import cv2
import torch
from ..tracker import Tracker
from .config import cfg_s50, cfg_st50, cfg_st101
from .models import Starks50, Starkst50, Starkst101
from .utils.stark_utils import Preprocessor, sample_target, merge_template_search, clip_box


class starks50(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'starks50')

        self.cfg = cfg_s50()

        self.network = Starks50().to(self.device).eval()

        self.preprocessor = Preprocessor()
        self.state = None

    def init(self, image, bbox):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, bbox, self.cfg.template_factor,
                                                    output_sz=self.cfg.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.state = bbox

    def track(self, image):
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.cfg.search_factor,
                                                                output_sz=self.cfg.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        print(pred_boxes)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.cfg.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
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


class starkst50(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True, update_intervals=200):
        super().__init__(checkpoint_path, use_cuda, 'starkst50')

        self.cfg = cfg_st50()

        self.network = Starkst50().to(self.device).eval()

        self.preprocessor = Preprocessor()
        self.state = None

        self.update_intervals = [update_intervals]
        self.num_extra_template = len(self.update_intervals)

    def init(self, image, bbox):
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, bbox, self.cfg.template_factor,
                                                      output_sz=self.cfg.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = bbox
        self.frame_id = 0

    def track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1

        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.cfg.search_factor,
                                                                output_sz=self.cfg.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.cfg.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.cfg.template_factor,
                                                            output_sz=self.cfg.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx + 1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        return self.state

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

class starkst101(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True, update_intervals=200):
        super().__init__(checkpoint_path, use_cuda, 'starkst101')

        self.cfg = cfg_st50()

        self.network = Starkst50().to(self.device).eval()

        self.preprocessor = Preprocessor()
        self.state = None

        self.update_intervals = [update_intervals]
        self.num_extra_template = len(self.update_intervals)

    def init(self, image, bbox):
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, bbox, self.cfg.template_factor,
                                                      output_sz=self.cfg.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = bbox
        self.frame_id = 0

    def track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1

        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.cfg.search_factor,
                                                                output_sz=self.cfg.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.cfg.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.cfg.template_factor,
                                                            output_sz=self.cfg.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx + 1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        return self.state

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
