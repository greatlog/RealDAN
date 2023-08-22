#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: degrader.py
# Created Date: Thursday August 17th 2022
# Author: Zhengxiong Luo
# Contact: <zhengxiong.luo@cripac.ia.ac.cn>
# 
# Last Modified: Tuesday August 22nd 2023 10:58:56 am
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
from functools import partial

from .module_util import ResBlockv2, ResidualBlockNoBN, Upsampler, default_init_weights
from utils.registry import ARCH_REGISTRY

class BasicBlock(nn.Sequential):
    def __init__(
        self, inc, outc, ksize, stride, bias=True, bn=False, act=nn.ReLU(True)
        ):

        m = [
            nn.Conv2d(inc, outc, ksize, stride, ksize//2, bias=bias)
        ]
        if bn:
            m.append(nn.BatchNorm2d(outc))
        if act is not None:
            m.append(act)
        super().__init__(*m)

class Estimator_S(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.fusion = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        self.body = nn.Sequential(
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, lrf, srf, degf):
        b, c, h, w = lrf.shape

        degf = degf.view(b, c, 1, 1).repeat(1, 1, h, w)
        f = torch.cat([lrf, srf, degf], dim=1)
        f = self.fusion(f)
        f = self.body(f)
        return f

class Estimator_L(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.fusion = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        self.body = nn.Sequential(
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1), ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1), ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1), nn.Conv2d(nf, nf, 3, 1, 1), 
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, lrf, srf, degf):
        b, c, h, w = lrf.shape

        degf = degf.view(b, c, 1, 1).repeat(1, 1, h, w)
        f = torch.cat([lrf, srf, degf], dim=1)
        f = self.fusion(f)
        f = self.body(f)
        return f


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class Restorer(nn.Module):
    def __init__(
        self, nf, nb, block="rrdb"
    ):
        super().__init__()

        self.fusion = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        act = partial(nn.LeakyReLU, 0.2, True)
        if block == "edsr":
            body = [ResidualBlockNoBN(num_feat=nf, res_scale=1, act=act) for _ in range(nb)]
        elif block == "rrdb":
            body = [RRDB(nf) for _ in range(nb)]
        self.body = nn.Sequential(*body)

    def forward(self, lrf, srf, degf):
        b, c, h, w = lrf.shape

        degf = degf.view(b, c, 1, 1).repeat(1, 1, h, w)

        f = torch.cat([lrf, srf, degf], dim=1)
        f = self.fusion(f)
        f = self.body(f) + f
       
        return f

@ARCH_REGISTRY.register()
class DAN(nn.Module):
    def __init__(
        self, nc, nf, nb, deg_dim=10, loop=3, scale=4, block="edsr", est_size="small"
    ):
        super().__init__()

        self.scale = scale
        self.loop = loop
        self.deg_dim = deg_dim

        self.img_head = nn.Conv2d(nc, nf, 3, 1, 1)
        self.deg_head = nn.Conv2d(deg_dim, nf, 1, 1, 0)

        self.Restorer = Restorer(nf, nb, block=block)
        self.Estimator = Estimator_S(nf) if est_size=="small" else Estimator_L(nf)

        self.img_tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            Upsampler(scale=scale, n_feat=nf),
            nn.Conv2d(nf, nc, 3, 1, 1)
        )

        self.deg_tail = nn.Sequential(
            nn.Conv2d(nf, deg_dim, 1, 1, 0)
        )

        self.init_deg = nn.Parameter(torch.zeros(1, deg_dim, 1, 1), requires_grad=True)

        self.register_buffer("mean", torch.FloatTensor([0.4488, 0.4371, 0.4040]))
        self.mean = self.mean.view(1, 3, 1, 1)

    def forward(self, lr):
        b, c, h, w = lr.shape

        lr = lr - self.mean

        lrf = self.img_head(lr)
        degf = self.deg_head(self.init_deg).repeat(b, 1, 1, 1)
        srf = lrf.clone()
        
        for i in range(self.loop):

            degf = self.Estimator(lrf, srf.detach(), degf) + degf
            srf = self.Restorer(lrf, srf, degf.detach()) + srf
        
        sr = self.img_tail(srf) + self.mean
        deg = self.deg_tail(degf).view(b, -1)

        return [sr], [deg]
