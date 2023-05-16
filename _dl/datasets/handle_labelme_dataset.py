import json
import os
import sys
from glob import glob
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

from bcommon import convert_coordinate as cc


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


def read(json_data):
    print("\n")
    categories = []
    print(data.keys())
    print(data['version'])
    print(data['flags'])
    for i in range(len(data['shapes'])):
        category_name = data['shapes'][i]['label']
        if category_name not in categories:
            categories.append(category_name)
        # print(data['categories'][i]['name'])

    print(categories)
    for i, c in enumerate(categories):
        print(f"  {i}: {c}")

    print(f"Image Width : {data['imageWidth']} | Image Height : {data['imageHeight']}")

    _ret = {
        "width": data['imageWidth'],
        "height": data['imageHeight'], 
        "polygon": data['shapes'],
        "category": categories,
        "filename": data['imagePath']
    }

    return _ret


def convert_dataset(data, savedir='', categories=[], dest='yolo'):
    width, height = data['width'], data['height']
    filename = data['filename']
    category_names = data['category'] if categories == [] else categories

    sentences = []

    if savedir != '' and not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)

    

    with open(os.path.join(savedir, filename.replace(".jpg", ".txt"))) as f:
        f.writelines(sentences)


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/Desktop/newdoughs'
    jsonlist = glob(os.path.join(dataset_rootpath, "*.json"))

    annotations = []

    # ======================= READ =======================
    for jsonfile in jsonlist:
        with open(jsonfile) as f:
            data = json.load(f)
        _annotation = read(data)
        annotations.append(_annotation)

    # ======================= READ =======================

    # ======================= WRITE =======================
    for annot in annotations:
        convert_dataset(annot, savedir=os.path.join(dataset_rootpath, 'txtfiles'), categories=['dough', 'tomato_sauce', 'mozzarella_cheese'])
    # ======================= WRITE =======================
