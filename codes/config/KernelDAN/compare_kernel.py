from glob import glob
import os
import os.path as osp
from scipy import io
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import filters, measurements, interpolation

import sys
sys.path.insert(0, "../../")
from utils import imresize, bgr2ycbcr


def kernel_shift(kernel, sf):
    current_center_of_mass = measurements.center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
    shift_vec = wanted_center_of_mass - current_center_of_mass
    return interpolation.shift(kernel, shift_vec)

def downscale(im, kernel, scale_factor, output_shape=None):

    if output_shape is None:
        output_shape = np.array(im.shape[:-1]) / np.array(scale_factor)

    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
    return imresize(out_im, 1/scale_factor[0])
    # return out_im[::scale_factor[0], ::scale_factor[1]]

def downscale_pt(im, kernel, scale_factor):
    x = torch.FloatTensor(im)[None].permute(0, 3, 1, 2).contiguous().cuda()
    k = torch.FloatTensor(kernel)[None, None].cuda()
    pad = (kernel.shape[0] // 2,) * 4
    x = F.pad(x,pad, mode="reflect")
    out = F.conv2d(x.transpose(0,1).contiguous(),k, stride=1).transpose(0,1).contiguous()
    out = imresize(out, 1 / scale_factor[0])
    return out[0].float().cpu().permute(1, 2, 0).contiguous().numpy()

def cal_psnr(im1, im2):
    mse = ((im1 - im2) ** 2).mean()
    return 10 * np.log10(1 / mse)

if __name__ == '__main__':
    gt_kernel_path = "/home/lzx/SRDatasets/DIV2KRK/gt_k_x4/"
    pt_kernel_path = "result/dan_kernelgan_x4_k31_kernel/DIV2KRK/"

    lr_img_path = "/home/lzx/SRDatasets/DIV2KRK/x4/"
    hr_img_path = "/home/lzx/SRDatasets/DIV2KRK/gt/"

    gt_kernels = sorted(glob(osp.join(gt_kernel_path, "*mat")))
    pt_kernels = sorted(glob(osp.join(pt_kernel_path, "*mat")))

    hr_images = sorted(glob(osp.join(hr_img_path, "*png")))
    lr_images = sorted(glob(osp.join(lr_img_path, "*png")))

    num_kernels = len(gt_kernels)

    kernel_error = 0
    img_psnr = 0
    for i in tqdm(range(num_kernels)):

        gt_kernel = io.loadmat(gt_kernels[i])["Kernel"]
        pt_kernel = io.loadmat(pt_kernels[i])["Kernel"]
        pt_kernel = pt_kernel / pt_kernel.sum()

        kernel_mse = ((gt_kernel/gt_kernel.max() - pt_kernel / pt_kernel.max()) ** 2).mean()
        kernel_error += kernel_mse

        hr = cv2.imread(hr_images[i]) / 255.0
        lr = cv2.imread(lr_images[i]) / 255.0

        # fake_lr = downscale(hr, pt_kernel, [4, 4])
        fake_lr = downscale_pt(hr, pt_kernel, [4, 4])
        psnr = cal_psnr(bgr2ycbcr(fake_lr, only_y=True), bgr2ycbcr(lr, only_y=True))
        img_psnr += psnr

    avg_error = kernel_error / num_kernels
    avg_psnr = img_psnr / num_kernels

    print(f"Average error: {avg_error}, Average PSNR: {avg_psnr}")
