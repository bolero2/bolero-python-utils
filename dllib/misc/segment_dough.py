import glob
import sys
sys.path.append("/home/bulgogi/bolero/AisttForGuideAI/sandbox/PIDNet")
import argparse
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
from time import time
from tabulate import tabulate
from copy import deepcopy
import shutil as sh
import models
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.append(os.getenv("PYTHON_UTILS"))
from bdataset import DatasetParser
from commons import get_colormap, image_blending
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
    weight_file = f"dough_segmentor/best.pt"

    model = models.pidnet.get_pred_model('pidnet-s', 2, activation='relu', dropout=False)

    model = load_pretrained(model, weight_file).cuda()
    model.eval()

    imglist = glob.glob(os.path.join("total", "images", "*.jpg"), recursive=True)

    image_size = (336, 192)
    sv_path = 'dough_result'

    if not os.path.isdir(sv_path):
        os.makedirs(sv_path, exist_ok=True)

    with torch.no_grad():
        for i, img_path in tqdm(enumerate(imglist), total=len(imglist)):
            img_name = os.path.basename(img_path)
            img = cv2.imread(os.path.abspath(img_path),
                             cv2.IMREAD_COLOR)
            ih, iw, ic = img.shape

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:],
            # pred = F.interpolate(pred, size=(ih, iw),
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            pred = pred.astype(np.uint8)

            sv_img = Image.fromarray(pred)
            sv_img = sv_img.resize((iw, ih)).convert('P')
            sv_img.putpalette(palette)
            savename = os.path.join(sv_path, img_name.replace('.jpg', '.png'))
            sv_img.save(savename)
