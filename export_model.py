# -*- coding: utf-8 -*-
# @Time : 2023/5/5 17:50
# @Author : zihua.zeng
# @File : test.py

import os
import torch
import cv2
import onnxruntime
import numpy as np
from basicsr.train import parse_options
from basicsr.models import create_model
from pprint import pprint


# 编写一个函数，他能读取 pytorch 模型，然后再转换成 onnx
def convert2onnx():
    pth_model_path = "experiments/edz/sidd/net_g_latest.pth"
    out_model_path = "denoise_nafnet_32.onnx"
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    model = create_model(opt).net_g
    weights_db = torch.load(pth_model_path, map_location="cpu")
    model.load_state_dict(weights_db['params'])
    model.eval()

    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},  # 这么写表示NCHW都会变化
    }

    torch.onnx.export(model,  # model being run
                      torch.randn((1, 3, 224, 224)),  # model input (or a tuple for multiple inputs)
                      out_model_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      dynamic_axes=dynamic_axes)


def preproc_image():
    image = cv2.imread("demo/test_denoise.png")
    test_image = cv2.resize(image, (512, 512))
    cv2.imwrite("demo/resize_denoise.png", test_image)
    pass


# 编写函数，他能读取图片和 onnx 模型，并执行推断
def inference():
    # 读取图片
    ori_img = cv2.imread('demo/resize_denoise.png')
    # ori_img = cv2.resize(ori_img, (256, 256))
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # 读取 onnx 模型
    sess = onnxruntime.InferenceSession('denoise_nafnet_32.onnx')

    # 执行推断
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([output_name], {input_name: img})[0]

    pred_onnx = (pred_onnx * 255.0).clip(0, 255)
    pred_onnx = np.transpose(pred_onnx, (0, 2, 3, 1))
    enhanced_images = np.squeeze(pred_onnx, axis=0)
    enhanced_images = enhanced_images.astype(np.uint8)
    enhanced_images = cv2.cvtColor(enhanced_images, cv2.COLOR_RGB2BGR)

    vs_image = np.hstack([ori_img, enhanced_images])
    cv2.imshow("vs_image", vs_image)
    cv2.waitKey(0)
    cv2.imwrite("demo/denoise_img.png", enhanced_images)


if __name__ == '__main__':
    # convert2onnx()
    # inference()
    preproc_image()
    inference()
    pass
