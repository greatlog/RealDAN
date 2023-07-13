import math
from collections import OrderedDict

import lpips as lp
import numpy as np
import torch
import pyiqa

from .ssim import calculate_ssim as ssim
from .best_psnr import best_psnr


class IQA:

    referecnce_metrics = ["best_psnr", "best_ssim", "psnr", "ssim", "lpips", "l1", "l2", "acc"]
    nonreference_metrics = ["niqe", "piqe", "brisque", "nrqm", "pi"]
    supported_metrics = referecnce_metrics + nonreference_metrics

    def __init__(
        self,
        metrics,
        lpips_type="alex",
        max_value=255,
        crop_border=0,
        cuda=True,
        y_channel=False,
    ):
        for metric in self.supported_metrics:
            if not (metric in self.supported_metrics):
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )
        self.max_value = max_value
        self.crop_border = crop_border if crop_border is not None  else 0
        self.cuda = cuda
        self.lpips_type = lpips_type

    def crop(self, inp):
        if self.crop_border != 0:
            pad = self.crop_border
            return inp[pad : -pad, pad : -pad]

    def __call__(self, res, ref=None, mask=None, metrics=["niqe"]):
        """
        res, ref: [0, max_value]
        """
        self.res = self.crop(res)
        self.res_t = torch.FloatTensor(self.res).unsqueeze(0)
        if self.res.ndim == 2:
            self.res_t = self.res_t.unsqueeze(-1)
        self.res_t = self.res_t.permute(0, 3, 1, 2).contiguous() / self.max_value
        if self.cuda:
            self.res_t = self.res_t.cuda()

        if ref is not None:
            self.ref = self.crop(ref)
            self.ref_t = torch.FloatTensor(self.ref).unsqueeze(0)
            if self.ref.ndim == 2:
                self.ref_t = self.ref_t.unsqueeze(-1)
            self.ref_t = self.ref_t.permute(0, 3, 1, 2).contiguous() / self.max_value
            if self.cuda:
                self.ref_t = self.ref_t.cuda()

        if mask is not None:
            self.mask = self.crop(mask)
            self.mask_t = torch.FloatTensor(self.mask).unsqueeze(0).unsqueeze(
                0).contiguous()
            if self.cuda:
                self.mask_t = self.mask_t.cuda()
        
        scores = OrderedDict()
        for metric in metrics:
            scores[metric] = getattr(self, f"calculate_{metric}")()
        if self.cuda:
            torch.cuda.empty_cache()
        return scores

    def calculate_lpips(self):
        if not hasattr(self, "lpips_fn"):
            self.lpips_fn = lp.LPIPS(net=self.lpips_type)
            if self.cuda:
                self.lpips_fn = self.lpips_fn.cuda()
        if not hasattr(self, "mask_t"):
            score = self.lpips_fn(self.res_t, self.ref_t, normalize=True)
        else:
            score = self.lpips_fn(
                self.res_t * self.mask_t,
                self.ref_t * self.mask_t,
                normalize=True
                )
        return score.item()

    def calculate_brisque(self):
        if not hasattr(self, "brisque_fn"):
            self.brisque_fn = pyiqa.create_metric("brisque")
            if self.cuda:
                self.brisque_fn = self.brisque_fn.cuda()
        score = self.brisque_fn(self.res_t)
        return score.item()

    def calculate_piqe(self):
        if not hasattr(self, "eng"):
            import matlab.engine
            self.eng = matlab.engine.start_matlab()
        if not hasattr(self, matlab_res):
            import matlab
            matlab_res = matlab.uint8(self.res.tolist())
        return self.eng.piqe(matlab_res)
    
    def calculate_niqe(self):
        if not hasattr(self, "niqe_fn"):
            self.niqe_fn = pyiqa.create_metric("niqe")
            if self.cuda:
                self.niqe_fn = self.niqe_fn.cuda()
        self.niqe_score = self.niqe_fn(self.res_t).item()
        return self.niqe_score
    
    # def calculate_niqe(self):
    #     if not hasattr(self, "eng"):
    #         import matlab.engine
    #         self.eng = matlab.engine.start_matlab()
    #     if not hasattr(self, matlab_res):
    #         import matlab
    #         matlab_res = matlab.uint8(self.res.tolist())
    #     self.niqe_score = self.eng.niqe(matlab_res)
    #     return self.niqe_score
    
    def calculate_nrqm(self):
        if not hasattr(self, "nrqm"):
            self.nrqm_fn = pyiqa.create_metric("nrqm")
            if self.cuda:
                self.nrqm_fn = self.nrqm_fn.cuda()
        score = self.nrqm_fn(self.res_t)
        self.nrqm_score = score.item()
        return self.nrqm_score
    
    def calculate_pi(self):
        if not hasattr(self, "nrqm_score"):
            self.calculate_nrqm()
        if not hasattr(self, "niqe_score"):
            self.calculate_niqe()
        return 0.5 * ((10 - self.nrqm_score) + self.niqe_score)
    
    # def calculate_pi(self):
    #     if not hasattr(self, "pi_fn"):
    #         self.pi_fn = pyiqa.create_metric("pi")
    #         if self.cuda:
    #             self.pi_fn = self.pi_fn.cuda()
    #     score = self.pi_fn(self.res_t)
    #     return score.item()

    def calculate_psnr(self):
        im1 = self.res.astype(np.float64)
        im2 = self.ref.astype(np.float64)
        mse = (im1 - im2) ** 2
        if hasattr(self, "mask"):
            mask = self.mask
            eps = 1e-12
            ratio = np.prod(mse.shape) / np.prod(mask.shape)
            if mse.ndim > 2:
                mask = mask[:, :, None]
            mse = (mse * mask).sum() / (mask.sum() * ratio + eps)
        else:
            mse = mse.mean()
        if mse == 0:
            return float("inf")
        else:
            return 20 * math.log10(self.max_value / math.sqrt(mse))

    def calculate_ssim(self):
        im1 = self.res.astype(np.float64)
        im2 = self.ref.astype(np.float64)
        ssim_maps = ssim(im1, im2, self.max_value)
        if hasattr(self, "mask"):
            mask = self.mask[5:-5, 5:-5]  # assume the window is 11x11
            eps = 1e-12
            ratio = np.prod(ssim_maps.shape) / np.prod(mask.shape)
            if ssim_maps.ndim > 2:
                mask = mask[:, :, None]
            s = (ssim_maps * mask).sum() / (mask.sum() * ratio + eps)
        else:
            s = ssim_maps.mean()
        return s
    
    def calculate_best_psnr(self):
        best_psnr_, best_ssim_ = best_psnr(self.res, self.ref, self.max_value)
        self.best_ssim = best_ssim_
        return best_psnr_
    
    def calculate_best_ssim(self):
        assert hasattr(self, "best_ssim")
        return self.best_ssim

    def calculate_l1(self):
        im1 = self.res.astype(np.float64)
        im2 = self.ref.astype(np.float64)

        diff = np.abs(im1 - im2)
        if hasattr(self, "mask"):
            diff = diff * self.mask
        return diff.mean()

    def calculate_l2(self):
        im1 = res.astype(np.float64)
        im2 = ref.astype(np.float64)

        diff = (im1 - im2) ** 2
        if hasattr(self, "mask"):
            diff = diff * self.mask
        return diff.mean()
    
    def calculate_acc(self):
        if self.res.ndim != self.ref.ndim:
            assert self.ref.ndim == self.res.ndim + 1
            res = np.expand_dims(self.res, -1)
        
        is_right = ((res - self.ref) == 0).sum().astype(np.float64)
        
        return is_right
