import json
import os
import sys
from glob import glob
from datetime import datetime
from PIL import Image
import cv2
import shutil as sh
import numpy as np
import copy
import argparse

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

from commons import convert_coordinate as cc
from handle_utils import read, labelme2yolo


"""
< LabelMe Dataset Format >
filename.json
  - version: string
  - flags: dictionary, {}
  - shapes: list contains dictionaries
    - label: string, label name (ex. doughs)
    - points: list in list, [[x1, y1], [x2, y2], [x3, y3], ...], abs coord, float
    - group_id: null
    - shape_type: "polygon"
    - flags: dictionary, {}
  - imagePath: string, filename.jpg
  - imageData: string, base64 encoded (not always in keys)
  - imageHeight: integer
  - imageWidth: integer
""" 

CATEGORIES = [] 


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('-d', '--dir', help='target directory name', default='')

    args = parser.parse_args()

    print("Args :", args, "\n\n\n")

    return args


if __name__ == "__main__":
    args = parse_args()
    pwd = os.getcwd()

    dirname = args.dir

    categories = ['dough', 'tomato_sauce', 'mozzarella_cheese', 'pepperoni', 'basil_oil', 'marinated_tomato', 'mayonnaise', 'gorgonzola', 'sweet_potato_mousse', 'onion', 'sweet_corn', 'better_bite', 'bacon', 'bulgogi_grinding', 'meat_sauce', 'pork_topping', 'roasted_onion_sauce', 'red_cheddar_cheese', 'jack_daniel_sauce', 'jack_daniel_and_mayonnaise', 'popcorn_chicken', 'cutting_corn', 'consomme_sauce', 'mushroom', 'bulgogi', 'green_pepper', 'black_olive']

    # dataset_rootpath = '/home/bulgogi/Desktop/3_total_sweet_potato/labelme'
    dataset_rootpath = '/home/bulgogi/Desktop/4_total_bacon_potato/labelme' if dirname == '' else dirname
    dataset_rootpath = os.path.abspath("./sample_images")
    print("[handle_labelme_dataset.py] dataset_rootpath :", dataset_rootpath)
    # dataset_rootpath = '/home/bulgogi/Desktop/sharing/20230629이전 요청 레이블링/이윤환'
    # dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ingredients/v1/dough/test'
    category = ''    # 'class_name' or ''

    jsonlist = glob(os.path.join(dataset_rootpath, "jsons", "*.json"))

    annotations = []

    # ======================= READ =======================
    for jsonfile in jsonlist:
        print("Target file :", jsonfile)
        imagefile = jsonfile.replace("/jsons/", "/images/").replace(".json", ".jpg")
        with open(jsonfile) as f:
            data = json.load(f)
        _annotation = read(data, imagefile, dtype='labelme')
        annotations.append(_annotation)
    # ======================= READ =======================

    annotations2 = copy.deepcopy(annotations)

    # ======================= WRITE =======================
    print("Write in 'total'")
    for annot in annotations:
        labelme2yolo(annot, savedir='labels', dataset_rootpath=dataset_rootpath, categories=categories, savetype='total')

    print("Write in 'ingredients'")
    for annot in annotations2:
        try:
            labelme2yolo(annot, savedir='labels', dataset_rootpath=dataset_rootpath, categories=categories, savetype='ingredients')
        except:
            continue
    # ======================= WRITE =======================
