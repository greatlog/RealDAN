#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: degrader.py
# Created Date: Thursday August 17th 2022
# Author: Zhengxiong Luo
# Contact: <zhengxiong.luo@cripac.ia.ac.cn>
# 
# Last Modified: Tuesday August 22nd 2023 10:55:19 am
# 
# Copyright (c) 2023 Center of Research on Intelligent Perception and Computing (CRIPAC)
# All rights reserved.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import random
from scipy import special
from collections import OrderedDict

from utils.registry import ARCH_REGISTRY
from utils.diffjpeg import DiffJPEG
from utils.deg_utils import generate_gaussian_noise_pt, generate_poisson_noise_pt


@ARCH_REGISTRY.register()
class Degrader(nn.Module):
    def __init__(
        self, blur_opt=None, noise_opt=None, resize_opt=None, jpeg_opt=None,
        resize_back_opt=None, deg_scale=1, crop_size=None, return_ratio=False
        ):
        super().__init__()

        if blur_opt is not None:
            self.blur_opt = blur_opt
            self.blur = RandomBlur(**blur_opt, return_ratio=return_ratio)
        
        if resize_opt is not None:
            self.resize_opt = resize_opt
            self.resize = RandomResize(**resize_opt, return_ratio=return_ratio)
        
        if noise_opt is not None:
            self.noise_opt = noise_opt
            self.noise = RandomNoise(**noise_opt, return_ratio=return_ratio)
        
        if resize_back_opt is not None:
            self.resize_back_opt = resize_back_opt
            self.resize_back = ResizeBack(**resize_back_opt, return_ratio=return_ratio)
        
        if jpeg_opt is not None:
            self.jpeg_opt = jpeg_opt
            self.jpeg = RandomJPEG(**jpeg_opt, return_ratio=return_ratio)
        
        self.deg_scale = deg_scale
        self.crop_size = crop_size
    
    def forward(self, x, src=None):
        src = x if src is None else src
        b, c, h, w = src.shape

        tgt_h = h // self.deg_scale
        tgt_w = w // self.deg_scale

        outs = OrderedDict()

        lr = x
        if hasattr(self, "blur_opt"):
            (
                lr, kernel,
                sigma_x, sigma_y, theta, beta,
                is_gauss, radius,
                is_sinc, omega
            ) = self.blur(lr)
            outs.update({
                # "kernel": kernel, 
                "sigma_x": sigma_x, "sigma_y": sigma_y,
                "theta": theta, "beta": beta, "is_gauss": is_gauss,
                "radius": radius, "is_sinc": is_sinc, "omega": omega
            })
            
        if hasattr(self, "resize_opt"):
            if self.resize_opt["base"] == "tgt":
                shape = (tgt_h, tgt_w)
            elif self.resize_opt["base"] == "src":
                shape = (h, w)
            lr, scale, resize_mode = self.resize(lr, shape=shape)
            outs.update({
                "scale": scale,
                "resize_mode": resize_mode
                })

        if hasattr(self, "noise_opt"):
            lr, noise, noise_type, color_type, gauss_level, poisson_level = self.noise(lr)
            outs.update({
                # "noise": noise,
                "noise_type": noise_type, "color_type": color_type,
                "gauss_level": gauss_level, "poisson_level": poisson_level
            })
        
        if hasattr(self, "resize_back"):
            lr, kernel, radius, is_sinc, omega, mode = self.resize_back(lr, shape=(tgt_h, tgt_w))
            outs.update({
                # "back_kernel": kernel,
                "back_radius": radius,
                "back_is_sinc": is_sinc,
                "back_omega": omega,
                "back_reize_mode": mode
            })
       
        if hasattr(self, "jpeg_opt"):
            lr, is_jpeg, jpeg_q = self.jpeg(lr)
            outs.update({
                "is_jpeg": is_jpeg,"jpeg_q": jpeg_q
            })

        lr = torch.clamp(lr * 255.0, 0, 255.0).round() / 255
        
        if self.crop_size is not None:
            if isinstance(self.crop_size, int):
                crop_size = self.crop_size
            elif isinstance(self.crop_size, list):
                crop_size = np.random.randint(self.crop_size[0], self.crop_size[1])

            rnd_h = np.random.randint(0, max(0, tgt_h - crop_size))
            rnd_w = np.random.randint(0, max(0, tgt_w - crop_size))

            lr = lr[:, :, rnd_h: rnd_h + crop_size, rnd_w: rnd_w + crop_size]

            rnd_h = int(rnd_h * self.scale); rnd_w = int(rnd_w * self.scale); crop_size = int(crop_size * self.scale)
            src = src[:, :, rnd_h: rnd_h + crop_size, rnd_w: rnd_w + crop_size]

        
        outs.update({
            "hr": src,
            "lr": lr
        })
        
        return outs

@ARCH_REGISTRY.register()
class RandomBlur(nn.Module):
    def __init__(self, 
        range_x, range_y=None, range_t=[-0.5, 0.5],
        range_betag=[0.5, 4], range_betap=[1, 2], range_radius=[3, 11],
        iso_prob=1, gaussian_prob=1, generalized_prob=0, sinc_prob=0,
        ksize=21, return_ratio=False
    ):
        super().__init__()

        self.ksize = ksize

        self.range_x = range_x; self.range_y = range_y; self.range_t = range_t
        self.range_betag = range_betag; self.range_betap = range_betap
        self.range_radius = range_radius

        self.sinc_prob = sinc_prob

        self.iso_prob = iso_prob; self.gaussian_prob = gaussian_prob
        self.generalized_prob = generalized_prob

        self.return_ratio = return_ratio

        ax = torch.arange(0, self.ksize) - self.ksize // 2
        xx, yy = torch.meshgrid(ax, ax)

        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
    
    def sample(self, srange):
        ratio = torch.rand(self.batch, device=self.device)
        value = ratio * (srange[1] - srange[0]) + srange[0]
        return ratio, value
    
    def get_ratio(self, srange, value):
        return (value - srange[0]) / (srange[1] - srange[0])
    
    def cal_distance(self, sigma_x, sigma_y, theta):
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        cos_theta_2 = cos_theta ** 2
        sin_theta_2 = sin_theta ** 2

        sigma_x_2 = 2.0 * (sigma_x ** 2)
        sigma_y_2 = 2.0 * (sigma_y ** 2)

        a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
        b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
        c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

        for tensor in [a, b, c]:
            tensor.unsqueeze_(-1).unsqueeze_(-1)

        fn = lambda x, y: a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2)
        distance = fn(
            self.xx.view(1, self.ksize, self.ksize),
            self.yy.view(1, self.ksize, self.ksize)
            )

        return distance
    
    def cal_j1(self, inp):
        out = special.j1(inp.detach().cpu().double().numpy())
        out = torch.from_numpy(out).to(self.device).float()
        return out
    
    def cal_sinc_kernel(self, radius):

        sinc_size = radius * 2 + 1
        omega_flag = (sinc_size < 13).float()
        min_omega = omega_flag * math.pi / 3 + (1 - omega_flag) * math.pi / 5; max_omega=math.pi

        ratio_omega = torch.rand(self.batch, device=self.device)
        omega = ratio_omega * (max_omega - min_omega) + min_omega

        dist = (self.xx ** 2 + self.yy ** 2).view(1, self.ksize, self.ksize)
        j1 = self.cal_j1(
            omega.view(self.batch, 1, 1) * dist
        )
        sinc_kernel = omega.view(self.batch, 1, 1) * j1 / (2 * math.pi * dist)

        sinc_kernel[:, self.ksize//2, self.ksize//2] = omega ** 2 / (4 * math.pi)

        return sinc_kernel, ratio_omega, omega

    def create_kernel(self):
        ratio_sigma_x, sigma_x = self.sample(self.range_x)
        ratio_sigma_y, sigma_y = self.sample(self.range_y)

        is_iso = torch.rand(self.batch, device=self.device)
        is_iso = (is_iso < self.iso_prob).float()

        ratio_sigma_y = is_iso * ratio_sigma_x + (1 - is_iso) * ratio_sigma_y
        sigma_y = is_iso * sigma_x + (1 - is_iso) * sigma_y

        ratio_theta, theta = self.sample(self.range_t)
        ratio_theta = is_iso * self.get_ratio(self.range_t, 0) + (1 - is_iso) * ratio_theta
        theta = (is_iso * 0 + (1 - is_iso) * theta) * math.pi

        distance = self.cal_distance(sigma_x, sigma_y, theta)
        
        ## caculate generalized gaussian
        is_gen = torch.rand(self.batch, device=self.device)
        is_gen = (is_gen < self.generalized_prob)
        
        ratio_betag, betag = self.sample(self.range_betag)
        betag = torch.where(is_gen, betag, betag.new_ones(betag.shape))
        ratio_betag = torch.where(
            is_gen, ratio_betag,
            betag.new_ones(betag.shape) * self.get_ratio(self.range_betag, 1)
            )
            
        ratio_betap, betap = self.sample(self.range_betap)

        gauss_kernel = torch.exp(-torch.float_power(distance, betag[:, None, None]))

        ## caculate plateau kernel
        plateau_kernel = 1 / (torch.float_power(distance, betap[:, None, None]) + 1)

        ## mix gauss kernels and plateau kernels
        is_gauss = torch.rand(self.batch, device=self.device)
        is_gauss = (is_gauss < self.gaussian_prob).float()
        gauss_flag = is_gauss[:, None, None]

        kernel = gauss_flag * gauss_kernel + (1 - gauss_flag) * plateau_kernel
        beta = is_gauss * betag + (1 - is_gauss) * betap
        ratio_beta = is_gauss * ratio_betag + (1 - is_gauss) * ratio_betap
       
        kernel = kernel.view(self.batch, self.ksize, self.ksize).float()

        ## caculate sinc kernel
        min_r, max_r = self.range_radius
        radius = torch.randint(
            low=min_r, high=max_r+1, size=(self.batch,)
            ).to(self.device)
        ratio_radius = self.get_ratio(self.range_radius, radius)

        sinc_kernel, ratio_omega, omega = self.cal_sinc_kernel(radius)
        
        is_sinc = torch.rand(self.batch, device=self.device)
        is_sinc = (is_sinc < self.sinc_prob).float()

        ## mix the sinc kernel with other kernels
        sinc_flag = is_sinc[:, None, None]
        kernel = sinc_kernel * sinc_flag + kernel * (1 - sinc_flag)
        omega = omega * is_sinc
        ratio_omega = self.get_ratio([0, math.pi], omega)

        for t in [sigma_x, sigma_y, theta, beta, ratio_beta, is_gauss]:
            t.mul_(1 - is_sinc)

        ratio_sigma_x = self.get_ratio(self.range_x, sigma_x)
        ratio_sigma_y = self.get_ratio(self.range_y, sigma_y)
        ratio_theta = self.get_ratio(self.range_t, theta/math.pi)

        ## cutoff the along the radius
        mask = ((self.xx.abs().unsqueeze(0) - radius.view(-1, 1, 1)) <= 0).float()
        mask = mask * ((self.yy.abs().unsqueeze(0) - radius.view(-1, 1, 1)) <= 0).float()
        kernel = kernel * mask

        kernel = kernel / torch.sum(kernel, (1, 2), keepdims=True)

        if self.return_ratio:
            return (
                kernel, ratio_sigma_x, ratio_sigma_y, ratio_theta, 
                ratio_beta, is_gauss, ratio_radius, is_sinc, ratio_omega
            )
        else:
            return (
                kernel, sigma_x, sigma_y, theta, beta, is_gauss,
                radius, is_sinc, omega
            )
    
    def forward(self, x, kernel=None):
        b, c, h, w = x.shape
        self.batch = b
        self.device = x.device
        
        if kernel is None:
            (
                kernel, sigma_x, sigma_y,
                theta, beta, is_gauss, radius,
                is_sinc, omega
            ) = self.create_kernel()

        pad = (self.ksize//2, ) * 4
        x = F.pad(x.view(b, c, h, w), pad, mode="reflect")
        x = x.transpose(0, 1).contiguous()

        x = F.conv2d(x, kernel.unsqueeze(1), groups=b, stride=1)
        x = x.transpose(0, 1).contiguous().view(b, c, *x.shape[2:])

        return x, kernel, sigma_x, sigma_y, theta, beta, is_gauss, radius, is_sinc, omega

@ARCH_REGISTRY.register()
class ResizeBack(nn.Module):
    def __init__(self, 
        ksize=21, range_radius=[3, 11], sinc_prob=1.0, 
        resize_mode=["area", "bicubic", "bilinear"], mode_prob=[0.33, 0.34, 0.33],
        return_ratio=False
    ):
        super().__init__()

        self.resize_mode = resize_mode
        self.mode_prob = mode_prob

        self.ksize = ksize
        self.range_radius = range_radius
        self.sinc_prob = sinc_prob
        self.return_ratio = return_ratio

        ax = torch.arange(0, self.ksize) - self.ksize // 2
        xx, yy = torch.meshgrid(ax, ax)

        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

        pulse_kernel = torch.zeros(ksize, ksize)
        pulse_kernel[ksize//2, ksize//2] = 1
        self.register_buffer("pulse_kernel", pulse_kernel)
    
    def sample_mode(self):
        resize_mode = np.random.choice(self.resize_mode, p=self.mode_prob)
        return resize_mode
    
    def sample(self, srange):
        ratio = torch.rand(self.batch, device=self.device)
        value = ratio * (srange[1] - srange[0]) + srange[0]
        return ratio, value
    
    def get_ratio(self, srange, value):
        return (value - srange[0]) / (srange[1] - srange[0])
    
    def cal_j1(self, inp):
        out = special.j1(inp.detach().cpu().double().numpy())
        out = torch.from_numpy(out).to(self.device).float()
        return out
    
    def cal_sinc_kernel(self, radius):

        sinc_size = radius * 2 + 1
        omega_flag = (sinc_size < 13).float()
        min_omega = omega_flag * math.pi / 3 + (1 - omega_flag) * math.pi / 5; max_omega=math.pi

        ratio_omega = torch.rand(self.batch, device=self.device)
        omega = ratio_omega * (max_omega - min_omega) + min_omega

        dist = (self.xx ** 2 + self.yy ** 2).view(1, self.ksize, self.ksize)
        j1 = self.cal_j1(
            omega.view(self.batch, 1, 1) * dist
        )
        sinc_kernel = omega.view(self.batch, 1, 1) * j1 / (2 * math.pi * dist)

        sinc_kernel[:, self.ksize//2, self.ksize//2] = omega ** 2 / (4 * math.pi)

        return sinc_kernel, ratio_omega, omega

    def create_kernel(self):
        ## caculate sinc kernel
        min_r, max_r = self.range_radius
        radius = torch.randint(
            low=min_r, high=max_r+1, size=(self.batch,)
            ).to(self.device)
        ratio_radius = self.get_ratio(self.range_radius, radius)

        sinc_kernel, ratio_omega, omega = self.cal_sinc_kernel(radius)
        
        is_sinc = torch.rand(self.batch, device=self.device)
        is_sinc = (is_sinc < self.sinc_prob).float()

        ## mix the sinc kernel with other kernels
        sinc_flag = is_sinc[:, None, None]
        kernel = sinc_kernel * sinc_flag + self.pulse_kernel[None] * (1 - sinc_flag)
        omega = omega * is_sinc
        ratio_omega = self.get_ratio([0, math.pi], omega)

        ## cutoff the along the radius
        mask = ((self.xx.abs().unsqueeze(0) - radius.view(-1, 1, 1)) <= 0).float()
        mask = mask * ((self.yy.abs().unsqueeze(0) - radius.view(-1, 1, 1)) <= 0).float()
        kernel = kernel * mask

        kernel = kernel / torch.sum(kernel, (1, 2), keepdims=True)

        if self.return_ratio:
            return (
                kernel, ratio_radius, is_sinc, ratio_omega
            )
        else:
            return (
                kernel, radius, is_sinc, omega
            )
    
    def forward(self, x, shape):
        b, c, h, w = x.shape
        self.batch = b
        self.device = x.device

        resize_mode = self.sample_mode()
        tgt_h, tgt_w = shape
        x = F.interpolate(x, size=(tgt_h, tgt_w), mode=resize_mode)

        mode = torch.zeros(b, len(self.resize_mode)).to(self.device)
        mode[:, self.resize_mode.index(resize_mode)] = 1
        
        kernel, radius, is_sinc, omega = self.create_kernel()

        pad = (self.ksize//2, ) * 4
        x = F.pad(x.view(b, c, tgt_h, tgt_w), pad, mode="reflect")
        x = x.transpose(0, 1).contiguous()

        x = F.conv2d(x, kernel.unsqueeze(1), groups=b, stride=1)
        x = x.transpose(0, 1).contiguous().view(b, c, *x.shape[2:])

        return x, kernel, radius, is_sinc, omega, mode

@ARCH_REGISTRY.register()
class RandomNoise:
    def __init__(self, range_g, range_p, gray_prob, gauss_prob, return_ratio=False):

        self.range_g = range_g
        self.range_p = range_p

        self.gray_prob = gray_prob
        self.gauss_prob = gauss_prob
        
        self.return_ratio = return_ratio
    
    def sample(self, srange):
        ratio = torch.rand(self.batch, device=self.device)
        value = ratio * (srange[1] - srange[0]) + srange[0]
        return ratio, value
    
    def get_ratio(self, srange, value):
        return (value - srange[0]) / (srange[1] - srange[0])
    
    def __call__(self, img):

        self.batch = img.shape[0]
        self.device = img.device

        color_type = torch.rand(self.batch, device=self.device)
        color_type = (color_type < self.gray_prob).float()

        ratio_p, sigma_p = self.sample(self.range_p)
        noise_p = generate_poisson_noise_pt(img, sigma_p, color_type)

        ratio_g, sigma_g = self.sample(self.range_g)
        noise_g = generate_gaussian_noise_pt(img, sigma_g, color_type)

        noise_type = torch.rand(self.batch, device=self.device)
        noise_type = (noise_type < self.gauss_prob).float()

        noise = noise_type[:, None, None, None] * noise_g + \
            (1 - noise_type[:, None, None, None]) * noise_p
        
        sigma_g = noise_type * sigma_g; ratio_g = self.get_ratio(self.range_g, sigma_g)
        sigma_p = (1 - noise_type) * sigma_p; ratio_p = self.get_ratio(self.range_p, sigma_p)
       
        img = img + noise.float()
        img = torch.clamp(img, 0, 1)
        
        if self.return_ratio:
            return img, noise, noise_type, color_type, ratio_g, ratio_p
        else:
            return img, noise, noise_type, color_type, sigma_g, sigma_p

@ARCH_REGISTRY.register()
class RandomResize:
    def __init__(
            self, range_scale, up_prob, down_prob,
            resize_mode, mode_prob, base, return_ratio
        ):
        
        self.range_scale = range_scale
        self.up_prob = up_prob
        self.down_prob = down_prob
        self.keep_prob = 1 - up_prob - down_prob

        self.resize_mode = resize_mode
        self.mode_prob = mode_prob

        self.return_ratio = return_ratio
        self.base = base
    
    def sample_scale(self):
        reszie_type = np.random.choice(
            ["up", "down", "keep"], 
            p=[self.up_prob, self.down_prob, self.keep_prob]
        )
        scale = random.random()
        if reszie_type == "up":
            scale = scale * (self.range_scale[1] - 1) + 1
        elif reszie_type == "down":
            scale = scale * (1 - self.range_scale[0]) + self.range_scale[0]
        elif reszie_type == "keep":
            scale = 1
        
        return scale
    
    def sample_mode(self):
        resize_mode = np.random.choice(self.resize_mode, p=self.mode_prob)
        return resize_mode
    
    def __call__(self, img, shape):

        scale = self.sample_scale()
        resize_mode = self.sample_mode()
        
        base_h, base_w = shape
        img = F.interpolate(img, size=(int(base_h*scale), int(base_w*scale)), mode=resize_mode)
        
        scale = torch.FloatTensor([scale]).to(img.device).repeat(img.shape[0])
        scale_ratio = (scale - self.range_scale[0]) / (self.range_scale[1] - self.range_scale[0])

        mode = scale.new_zeros(img.shape[0], len(self.resize_mode))
        mode[:, self.resize_mode.index(resize_mode)] = 1

        if self.return_ratio:
            return img, scale_ratio, mode
        else:
            return img, scale, mode

@ARCH_REGISTRY.register()
class RandomJPEG(nn.Module):
    def __init__(self, range_q, diff=False, jpeg_prob=0.5, return_ratio=False):
        super().__init__()

        self.quality_range = range_q
        self.return_ratio = return_ratio
        self.jpeg_prob = jpeg_prob

        self.jpeg = DiffJPEG(differentiable=False)
    
    def forward(self, img):

        is_jpeg = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        is_jpeg = (is_jpeg < self.jpeg_prob).float()

        min_q, max_q = self.quality_range
        ratio = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        quality = ratio * (max_q - min_q) + min_q

        img = torch.clamp(img, 0, 1)
        jpeg_img = self.jpeg(img, quality.clone())

        x = is_jpeg[:, None, None, None] * jpeg_img + (1 - is_jpeg[:, None, None, None]) * img
        quality = is_jpeg * quality + (1 - is_jpeg) * (100)
        ratio = (quality - min_q) / (max_q - min_q)

        if self.return_ratio:
            return x, is_jpeg, ratio
        else:
            return x, is_jpeg, quality

