#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: degrader.py
# Created Date: Thursday August 17th 2022
# Author: Zhengxiong Luo
# Contact: <zhengxiong.luo@cripac.ia.ac.cn>
# 
# Last Modified: Tuesday August 22nd 2023 10:56:21 am
# 
# Copyright (c) 2023 Center of Research on Intelligent Perception and Computing (CRIPAC)
# All rights reserved.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###


import argparse
import sys
from time import time
from tqdm import tqdm
import torch
from torchsummaryX import summary

sys.path.append("../../")
import utils.option as option
from models import create_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to option YMAL file of Predictor.",
)
args = parser.parse_args()
opt = option.parse(args.opt, root_path=".", is_train=False)

opt = option.dict_to_nonedict(opt)
model = create_model(opt)

test_tensor = torch.randn(1, 3, 270, 180).cuda()
start = time()
for idx in tqdm(range(1000)):
    out = model.netSR(test_tensor)
duration = time() - start
speed = duration / 1000
print(f"Speed {speed} s/image")

