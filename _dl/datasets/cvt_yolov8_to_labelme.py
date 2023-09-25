import json
import os
import sys
from glob import glob
from datetime import datetime
from PIL import Image
import cv2
import shutil as sh
import numpy as np

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

from bcommon import convert_coordinate as cc
from handle_utils import read


"""
< LabelMe Dataset Format >
filename.json
  - version: string
  - flags: dictionary, {}
  - shapes: list contains dictionaries
    - label: string, label name (ex. dough)
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


if __name__ == "__main__":
    # categories = ['dough', 'tomato_sauce', 'mozzarella_cheese', 'pepperoni', 'basil_oil', 'marinated_tomato', 'mayonnaise', 'gorgonzola', 'sweet_potato_mousse', 'onion', 'sweet_corn', 'better_bite', 'bacon', 'bulgogi_grinding']
    # categories = ['bulgogi', 'green_pepper', 'mushroom', 'onion', 'roasted_onion_sauce']
    categories = ['bacon', 'better_bite', 'bulgogi_grinding', 'mozzarella_cheese', 'onion', 'tomato_sauce']

    # dataset_rootpath = f'/home/bulgogi/bolero/dataset/aistt_ingredients/v2/green_pepper'
    dataset_rootpath = f'/home/bulgogi/Desktop/sharing/_finished/sungho-baconpotato-finished-checking/labeled'
    category = ''    # 'class_name' or ''

    txtlist = glob(os.path.join(dataset_rootpath, "labels", "*.txt"))

    annotations = {}

    # ======================= READ =======================
    for txtfile in txtlist:
        print("Target file :", txtfile)
        imagefile = txtfile.replace("/labels/", "/images/").replace(".txt", ".jpg")
        with open(txtfile) as f:
            lines = f.readlines()
        _annotation = read(lines, imagefile, dtype='yolo', classes=category if category != '' else categories)
        annotations[imagefile] = _annotation
    # ======================= READ =======================
    os.makedirs(os.path.join(dataset_rootpath, "jsons"), exist_ok=True)

    # ======================= WRITE =======================
    for imgfile, _dict in annotations.items():
        basename = os.path.basename(imgfile).replace(".jpg", ".json")
        json_string = json.dumps(_dict)
        with open(os.path.join(dataset_rootpath, 'jsons', basename), 'w') as f:
            json.dump(_dict, f, indent=4)
    # ======================= WRITE =======================
