import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy import io
import math

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from metrics import IQA
from models import create_model
from utils import bgr2ycbcr, imresize


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--opt", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "--root_path",
        help="experiment configure file name",
        default="../../../",
        type=str,
    )
    # distributed training
    parser.add_argument("--gpu", help="gpu id for multiprocessing training", type=str)
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    opt["dist"] = args.world_size > 1

    util.mkdirs(
        (path for key, path in opt["path"].items() if not key == "experiments_root")
    )

    os.system("rm ./result")
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

    if opt["dist"]:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt, args))
    else:
        main_worker(0, 1, opt, args)


def main_worker(gpu, ngpus_per_node, opt, args):

    if opt["dist"]:
        if args.dist_url == "env://" and args.rank == -1:
            rank = int(os.environ["RANK"])

        rank = args.rank * ngpus_per_node + gpu
        print(
            f"Init process group: dist_url: {args.dist_url}, world_size: {args.world_size}, rank: {rank}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )

        torch.cuda.set_device(gpu)

    else:
        rank = 0

    torch.backends.cudnn.benchmark = True

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO,
        screen=True,
        tofile=True,
    )

    measure = IQA(**opt["evaluation"])

    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_datasets = []
    test_loaders = []

    for phase, dataset_opt in sorted(opt["datasets"].items()):

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt["dist"])

        if rank == 0:
            logger.info(
                "Number of test images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(test_set)
                )
            )
        test_datasets.append(test_set)
        test_loaders.append(test_loader)

    # load pretrained model by default
    model = create_model(opt)

    for test_dataset, test_loader in zip(test_datasets, test_loaders):

        test_set_name = test_dataset.opt["name"]
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)

        if rank == 0:
            logger.info("\nTesting [{:s}]...".format(test_set_name))
            util.mkdir(dataset_dir)

        validate(
            model,
            test_dataset,
            test_loader,
            opt,
            measure,
            dataset_dir,
            test_set_name,
            logger,
        )


def validate(
    model, dataset, dist_loader, opt, measure, dataset_dir, test_set_name, logger
):

    test_results = {}
    eval_opt = opt["evaluation"]
    if eval_opt["y_channel"]:
        test_results_y = {}
    for metric in eval_opt["metrics"]:
        test_results[metric] = torch.zeros((len(dataset))).cuda()
        if eval_opt["y_channel"]:
            test_results_y[metric] = torch.zeros((len(dataset))).cuda()

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    indices = list(range(rank, len(dataset), world_size))
    for (
        idx,
        test_data,
    ) in enumerate(dist_loader):
        idx = indices[idx]

        img_path = test_data["src_path"][0]
        img_name = img_path.split("/")[-1].split(".")[0]

        model.test(test_data)
        visuals = model.get_current_visuals()
        deg = visuals["deg"].numpy()

        sigma_x = deg[0] * (5 - 0.6) + 0.6
        sigma_y = deg[1] * (5 - 0.6) + 0.6
        theta = deg[2] * 0.5 * math.pi

        ax = np.arange(0, 31) - 31 // 2
        xx, yy = np.meshgrid(ax, ax)

        cos_theta = np.cos(theta); sin_theta = np.sin(theta)
        cos_theta_2 = cos_theta ** 2; sin_theta_2 = sin_theta ** 2
        sigma_x_2 = 2 * sigma_x ** 2; sigma_y_2 = 2 * sigma_y ** 2

        a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
        b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
        c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

        fn = lambda x, y: a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2)
        
        kernel = np.exp(-fn(xx, yy))
        print(kernel.shape)

        if opt["save"]:
            suffix = opt["suffix"]
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + ".mat")
            else:
                save_img_path = os.path.join(dataset_dir, img_name + ".mat")
            io.savemat(save_img_path, {"Kernel": kernel})

        print(save_img_path)

if __name__ == "__main__":
    main()
