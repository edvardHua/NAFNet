# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os

import cv2
import numpy as np
import torch

from tqdm import tqdm
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite


# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str


def proc_file(opt, img_path):
    output_path = opt['img_path'].get('output_img')
    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished. saved to {output_path}')


def proc_dir(opt, img_path):
    output_path = opt['img_path'].get('output_img')
    os.makedirs(output_path, exist_ok=True)
    file_client = FileClient('disk')
    model = create_model(opt)
    for f in tqdm(os.listdir(img_path)):
        suffix = f.split('.')[-1]
        if suffix.lower() not in ["jpg", "png"]:
            continue
        img_bytes = file_client.get(os.path.join(img_path, f), None)
        try:
            ori_img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        img = img2tensor(ori_img, bgr2rgb=True, float32=True)

        ## 2. run inference
        opt['dist'] = False

        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])

        h, w, _ = ori_img.shape
        sr_img = cv2.resize(sr_img, (w, h))
        union = np.hstack([(ori_img * 255.).astype(np.uint8), sr_img])
        imwrite(union, os.path.join(output_path, f.replace(".jpg", ".png")))


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_path = opt['img_path'].get('input_img')

    if os.path.isdir(img_path):
        proc_dir(opt, img_path)
    elif os.path.isfile(img_path):
        proc_file(opt, img_path)


if __name__ == '__main__':
    main()
