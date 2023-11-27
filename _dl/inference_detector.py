import glob
import sys
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
from commons import get_colormap, image_blending, convert_coordinate
from metrics import get_metric_for_segmentation
import getpass

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = get_colormap(255)
palette = [value for color in color_map for value in color]


if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if is_cuda else "cpu")

    dough_weight_file = '/home/bulgogi/bolero/projects/20221229-4f323b486b37774e/20221229-4f323b486b37774e.json'
    pepperoni_weight_file = "/home/bulgogi/bolero/projects/20221111-1e45ebe4995f53a5/20221111-1e45ebe4995f53a5.json"

    finder = Finder(dough_weight_file, pepperoni_weight_file, None, is_cuda, _device)

    imglist = glob.glob(os.path.join("/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/new_total_det_dataset/new", "**", "*.jpg"), recursive=True)
    print(imglist)

    image_size = (336, 192)
    sv_path = '/home/bulgogi/labeler/1'

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

            det_result = finder.inference_detection(img, include_labels=[0])
            
            sentences = []
            for d in det_result:
                _, xmin, ymin, xmax, ymax, conf = d
                if conf < 0.7:
                    continue
                else:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

                    get_ccwh = convert_coordinate([xmin, ymin, xmax, ymax], (ih, iw), 'xyrb', 'abs', 'ccwh', 'relat')
                    coord = get_ccwh()

                    classnum = 1
                    sentence = f"{classnum} {coord[0]} {coord[1]} {coord[2]} {coord[3]}\n"

                    sentences.append(sentence)

            cv2.imshow("img", img)
            key = cv2.waitKey(1)

            if key == ord('q'):
                exit()

            if sentences != []:
                annot_file = os.path.abspath(img_path).replace(".jpg", ".txt")
                f = open(annot_file, 'w')
                f.writelines(sentences)
                f.close()

                sh.copy(os.path.abspath(img_path), sv_path)
                sh.copy(os.path.abspath(annot_file), sv_path)
