import torch
from .config import cfg
from .models import LightFC
from ..tracker import Tracker
from .utils import sample_target, Preprocessor, hann2d, clip_box


class lightfc(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'lightfc')

        self.cfg = cfg()
        self.network = LightFC().to(self.device).eval()
        self.load_checkpoint()

        for module in self.network.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.search_size // self.cfg.stride

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

    def init(self, image, bbox):
        H, W, _ = image.shape

        z_patch_arr, resize_factor = sample_target(image, bbox, self.cfg.template_factor,
                                                   output_sz=self.cfg.template_size)

        template = self.preprocessor.process(z_patch_arr)

        with torch.no_grad():
            self.zf = self.network.forward_backbone(template)

        self.state = bbox
        self.frame_id = 0

    def track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.cfg.search_factor,
                                                   output_sz=self.cfg.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            out_dict = self.network.forward_tracking(z_feat=self.zf, x=search)

        response_origin = self.output_window * out_dict['score_map']

        pred_box_origin = self.compute_box(response_origin, out_dict,
                                           resize_factor).tolist()  # .unsqueeze(dim=0)  # tolist()

        self.state = clip_box(self.map_box_back(pred_box_origin, resize_factor), H, W, margin=2)

        return self.state

    def compute_box(self, response, out_dict, resize_factor):
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = (pred_boxes.mean(dim=0) * self.cfg.search_size / resize_factor)
        return pred_boxes

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
