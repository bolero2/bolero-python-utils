import glob
import sys
# sys.path.append("/home/bulgogi/bolero/AisttForGuideAI/sandbox/PIDNet")
sys.path.append("/home/bulgogi/bolero/AisttForGuideAI/AISTT/aistt_utils")
from Finder import Finder
import argparse
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
from time import time
import copy
from tabulate import tabulate
from copy import deepcopy
import shutil as sh
import models
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.append(os.getenv("PYTHON_UTILS"))
from bdataset import DatasetParser
from bcommon import get_colormap, image_blending
from metrics import get_metric_for_segmentation
import getpass

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = get_colormap(255)
palette = [value for color in color_map for value in color]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std 
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model


if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if is_cuda else "cpu")

    seg_weight_file = f"dough_segmentor/best.pt"
    dough_weight_file = "/home/bulgogi/bolero/projects/c9e330a8fdb11960/c9e330a8fdb11960.json"

    finder = Finder(dough_weight_file, None, seg_weight_file, is_cuda, _device)

    """
    model = models.pidnet.get_pred_model('pidnet-s', 2, activation='relu', dropout=False)

    model = load_pretrained(model, weight_file).cuda()
    model.eval()
    """

    imglist = glob.glob(os.path.join("total", "*.jpg"), recursive=True)

    image_size = (336, 192)
    sv_path = 'dough_result'

    if os.path.isdir(sv_path):
        sh.rmtree(sv_path)

    if not os.path.isdir(sv_path):
        os.makedirs(sv_path, exist_ok=True)

    with torch.no_grad():
        for i, img_path in tqdm(enumerate(imglist), total=len(imglist)):
            img_name = os.path.basename(img_path)
            img = cv2.imread(os.path.abspath(img_path),
                             cv2.IMREAD_COLOR)
            ori_img = copy.deepcopy(img)
            ih, iw, ic = img.shape

            full_canvas = np.zeros((ih, iw)).astype(np.uint8)

            _, dough_info, no_margin_coord = finder.find_dough(img)
            for elem in zip(dough_info, no_margin_coord):
                if elem[0][-1] < 0.8:
                    continue

                xmin, ymin, xmax, ymax = elem[0][1:-1]
                img = ori_img[ymin:ymax, xmin:xmax, :]
                print(">>> dough shape :", img.shape)
                dh, dw, dc = img.shape

                cv2.imshow("dough", img)

                img = cv2.resize(img, image_size)
                sv_img = np.zeros_like(img).astype(np.uint8)
                img = input_transform(img)
                img = img.transpose((2, 0, 1)).copy()
                img = torch.from_numpy(img).unsqueeze(0).cuda()
                pred = finder.segmodel(img)
                pred = F.interpolate(pred, size=(dh, dw),
                # pred = F.interpolate(pred, size=(ih, iw),
                                     mode='bilinear', align_corners=True)
                pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                pred = pred.astype(np.uint8)
                print('Pred\n', pred)
                print("pred shape :", pred.shape)

                """
                sv_img = Image.fromarray(pred)
                sv_img = sv_img.resize((iw, ih)).convert('P')
                sv_img.putpalette(palette)
                sv_img = np.array(sv_img)
                """
                full_canvas[ymin:ymax, xmin:xmax] = pred
                # savename = os.path.join(sv_path, img_name.replace('.jpg', '.png'))
                # sv_img.save(savename)

            full_canvas = Image.fromarray(full_canvas).convert('P')
            full_canvas.putpalette(palette)
            full_canvas.save(os.path.join(sv_path, img_name.replace('.jpg', '.png')))
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
