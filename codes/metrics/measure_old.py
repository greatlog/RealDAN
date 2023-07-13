import math
from collections import OrderedDict

import lpips as lp
import numpy as np

from .ssim import calculate_ssim as ssim
from .best_psnr import best_psnr


class IQA:

    referecnce_metrics = ["best_psnr", "best_ssim", "psnr", "ssim", "lpips", "l1", "l2", "acc"]
    nonreference_metrics = ["niqe", "piqe", "brisque"]
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
        self.crop_border = crop_border

        if "lpips" in metrics:
            self.lpips_fn = lp.LPIPS(net=lpips_type)
            self.cuda = cuda
            if cuda:
                self.lpips_fn = self.lpips_fn.cuda()
        if ("niqe" in metrics) or ("piqe" in metrics) or ("brisque" in metrics):
            import matlab.engine

            print("Starting matlab engine ...")
            self.eng = matlab.engine.start_matlab()

    def __call__(self, res, ref=None, mask=None, metrics=["niqe"]):
        """
        res, ref: [0, max_value]
        """
        if self.crop_border != 0 and self.crop_border is not None:
            if res is not None:
                res = res[
                    self.crop_border : -self.crop_border,
                    self.crop_border : -self.crop_border,
                ]
            if ref is not None:
                ref = ref[
                    self.crop_border : -self.crop_border,
                    self.crop_border : -self.crop_border,
                ]

        if hasattr(self, "eng"):
            import matlab

            self.matlab_res = matlab.uint8(res.tolist())

        scores = OrderedDict()
        for metric in metrics:
            if metric in self.referecnce_metrics:
                if ref is None:
                    raise ValueError(
                        "Ground-truth refernce is needed for {}".format(metric)
                    )
                scores[metric] = getattr(self, "calculate_{}".format(metric))(
                    res, ref, mask
                )

            elif metric in self.nonreference_metrics:
                scores[metric] = getattr(self, "calculate_{}".format(metric))(res)

            else:
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )
        return scores

    def calculate_lpips(self, res, ref, mask):
        if res.ndim < 3:
            return 0
        res = lp.im2tensor(res, factor=self.max_value, cent=0)
        ref = lp.im2tensor(ref, factor=self.max_value, cent=0)
        if self.cuda:
            res = res.cuda()
            ref = ref.cuda()
        score = self.lpips_fn(res, ref)
        return score.item()

    def calculate_niqe(self, res):
        return self.eng.niqe(self.matlab_res)

    def calculate_brisque(self, res):
        return self.eng.brisque(self.matlab_res)

    def calculate_piqe(self, res):
        return self.eng.piqe(self.matlab_res)

    def calculate_psnr(self, res, ref, mask):
        im1 = res.astype(np.float64)
        im2 = ref.astype(np.float64)
        mse = (im1 - im2) ** 2
        if mask is not None:
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

    def calculate_ssim(self, res, ref, mask):
        im1 = res.astype(np.float64)
        im2 = ref.astype(np.float64)
        ssim_maps = ssim(im1, im2, self.max_value)
        if mask is not None:
            mask = mask[5:-5, 5:-5]  # assume the window is 11x11
            eps = 1e-12
            ratio = np.prod(ssim_maps.shape) / np.prod(mask.shape)
            if ssim_maps.ndim > 2:
                mask = mask[:, :, None]
            s = (ssim_maps * mask).sum() / (mask.sum() * ratio + eps)
        else:
            s = ssim_maps.mean()
        return s
    
    def calculate_best_psnr(self, res, ref, mask):
        best_psnr_, best_ssim_ = best_psnr(res, ref, self.max_value)
        self.best_ssim = best_ssim_
        return best_psnr_
    
    def calculate_best_ssim(self, res, ref, mask):
        assert hasattr(self, "best_ssim")
        return self.best_ssim

    def calculate_l1(self, res, ref, mask):
        im1 = res.astype(np.float64)
        im2 = ref.astype(np.float64)

        diff = np.abs(im1 - im2)
        if mask is not None:
            diff = diff * mask
        return diff.mean()

    def calculate_l2(self, res, ref, mask):
        im1 = res.astype(np.float64)
        im2 = ref.astype(np.float64)

        diff = (im1 - im2) ** 2
        if mask is not None:
            diff = diff * mask
        return diff.mean()
    
    def calculate_acc(self, res, ref):
        if res.ndim != ref.ndim:
            assert ref.ndim == res.ndim + 1
            res = np.expand_dims(res, -1)
        
        is_right = ((res - ref) == 0).sum().astype(np.float64)
        
        return is_right
