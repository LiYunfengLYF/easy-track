import cv2
import torch
import numpy as np
from .config import cfg
from .models import SiamFC
from ..tracker import Tracker
from .utils import crop_and_resize


class siamfc(Tracker):
    def __init__(self, checkpoint_path=None, use_cuda=True):
        super().__init__(checkpoint_path, use_cuda, 'siamfc')
        self.cfg = cfg()

        self.network = SiamFC().to(self.device).eval()
        self.load_checkpoint()

    @torch.no_grad()
    def init(self, image: np.array, bbox: list) -> None:
        # convert box to 0-indexed and center based [y, x, h, w]
        bbox = np.array([bbox[1] - 1 + (bbox[3] - 1) / 2, bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[3], bbox[2]],
                        dtype=np.float32)
        self.center, self.target_sz = bbox[:2], bbox[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        z = crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.network.backbone(z)

    @torch.no_grad()
    def track(self, image: np.array) -> list:
        # search images
        x = [crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        x = self.network.backbone(x)
        responses = self.network.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = [float(self.center[1] + 1 - (self.target_sz[1] - 1) / 2),
               float(self.center[0] + 1 - (self.target_sz[0] - 1) / 2),
               float(self.target_sz[1]),
               float(self.target_sz[0])]

        return box
