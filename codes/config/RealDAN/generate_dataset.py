import argparse
import cv2
import os
import sys
import pickle
import lmdb
import glob
import torch
import shutil
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

sys.path.append("../../")
import utils as util
import utils.option as option
from archs import build_network
from utils import ProgressBar


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    img_save_path = opt["img_save_path"]
    if not osp.exists(img_save_path):
        util.mkdir(img_save_path)
    
    if opt["generate_lr_imgs"]:
        print("Generating lr images ...")

        netDeg1 = build_network(opt["netDeg1"]).cuda()
        netDeg2 = build_network(opt["netDeg2"]).cuda()

        degradations = {}

        src_paths = sorted(glob.glob(osp.join(opt["dataroot"], "*png")))

        for src_path in tqdm(src_paths):
            img_name = src_path.split("/")[-1].split(".")[0]

            sr_img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            sr_img = util.img2tensor(sr_img).unsqueeze(0).cuda()

            with torch.no_grad():
                deg_outs1 = netDeg1(sr_img)
                deg_outs2 = netDeg2(deg_outs1["lr"], sr_img)

            deg_outs = deg_outs1
            for k, v in deg_outs2.items():
                if not(k in ["lr", "hr"]):
                    if k in deg_outs.keys():
                        deg_outs[k] = torch.stack(
                            [deg_outs[k], v], -1
                        )
                    else:
                        deg_outs[k] = v
            
            lr = deg_outs2.pop("lr")
            hr = deg_outs2.pop("hr")

            lr_img = util.tensor2img(lr[0].detach().cpu())
            cv2.imwrite(osp.join(img_save_path, f"{img_name}.png"), lr_img)

            degradations[img_name] = OrderedDict()
            for k, v in deg_outs.items():
                if v is not None:
                    degradations[img_name][k] = v.squeeze().cpu().numpy()
        
        pickle.dump(degradations, open(osp.join(img_save_path, "degradations.pkl"), "wb"))
    
    if opt["create_lmdb"]:
        print("creating lmdb")

        batch = 1000

        lmdb_save_path = opt["lmdb_save_path"]
        if osp.exists(lmdb_save_path):
            shutil.rmtree(lmdb_save_path)

        img_list = sorted(glob.glob(osp.join(img_save_path, "*png")))

        data_size = sum(os.stat(v).st_size for v in img_list)
        key_l = []
        resolution_l = []
        pbar = ProgressBar(len(img_list))
        env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
        txn = env.begin(write=True)  # txn is a Transaction object

        for i, v in enumerate(img_list):
            pbar.update("Write {}".format(v))
            base_name = osp.splitext(osp.basename(v))[0]
            key = base_name.encode("ascii")
            data = cv2.imread(v, cv2.IMREAD_UNCHANGED)
            if data.ndim == 2:
                H, W = data.shape
                C = 1
            else:
                H, W, C = data.shape
            txn.put(key, data)
            key_l.append(base_name)
            resolution_l.append("{:d}_{:d}_{:d}".format(C, H, W))
            # commit in mode 2
            if i % batch == 1:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        env.close()

        print("Finish writing lmdb.")
        meta_info = {}
        #### create meta information
        # check whether all the images are the same size
        same_resolution = len(set(resolution_l)) <= 1
        if same_resolution:
            meta_info["resolution"] = [resolution_l[0]]
            meta_info["keys"] = key_l
            print("All images have the same resolution. Simplify the meta info...")
        else:
            meta_info["resolution"] = resolution_l
            meta_info["keys"] = key_l
            print("Not all images have the same resolution. Save meta info for each image...")

        #### pickle dump
        pickle.dump(meta_info, open(osp.join(lmdb_save_path, "meta_info.pkl"), "wb"))
        print("Finish creating lmdb meta info.")

if __name__ == "__main__":
    main()
