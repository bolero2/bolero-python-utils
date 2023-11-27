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

from commons import convert_coordinate as cc


"""
< COCO Dataset Format >
coco.json
  - info: dictionary
    - description: string
    - url: string
    - version: string
    - year: int
    - contributor: string
    - date_created: string

  - licenses: list contains dictionaries
    - url: string
    - id: int
    - name: string

  - images: list contains dictionaries
    - license: string
    - file_name: string
    - coco_url: string
    - height: int
    - width: int
    - date_captured: string
    - flickr_url: string
    - id: int

  - annotations: list contains dictionaries
    - segmentation: list in list, [[x1, y1, x2, y2, x3, y3, ...]]
    - area: int
    - is_crowd: 0
    - image_id: int, this value have to match with "images" - "id"
    - bbox: list, [x1, y1, x2, y2], xywh(abs)
    - category_id: int
    - id: int

  - categories: list contains dictionaries
    - supercategory: string
    - id: int, this value is specific and started from 1 (not 0, 0 should be supercategory)
    - name: string
""" 


def COCO_DEFAULT_FORMAT():
    _DICT = {}

    _DICT['info'] = {
        "description": "GOPIZZA CUSTOM DATASET",
        "url": "",
        "version": "1.0",
        "year": 2023,
        "contributor": "GOPIZZA-FutureLabs-bolero",
        "date_created": str(datetime.utcnow()).split(' ')[0].replace("-", "/"),
    }

    _DICT['licenses'] = []
    _LICENSE = {
        "url": "",
        "id": 1,
        "name": "NONE"
    }
    _DICT['licenses'].append(_LICENSE)

    _DICT['images'] = []
    _DICT['annotations'] = []
    _DICT['categories'] = []

    return _DICT


def read(json_data):
    # print(data)
    categories = []
    print(data.keys())
    print(data['info'].keys())
    print(data['images'][0].keys())
    print(data['annotations'][0].keys())
    for i in range(len(data['categories'])):
        category_name = data['categories'][i]['name']
        categories.append(category_name)
        # print(data['categories'][i]['name'])

    print(categories)
    for i, c in enumerate(categories):
        print(f"  {i}: {c}")


def make_json_dict(imglist, annotlist, category, fromtype='yolo'):
    """
    1. checking this github repository: https://github.com/ultralytics/JSON2YOLO
    2. there is predownloaded repository in local machine: /home/{username}/bolero/JSON2YOLO
    """
    _CATEGORIES = []
    """
    _CATEGORIES.append({
        "supercategory": "none",
        "id": 0,
        "name": "pizza_ingredient"
    })
    """

    if isinstance(category, str) and category.split('.')[-1] == 'txt':
        category = open(category, 'r').readlines()

    for i, c in enumerate(category):
        c = c.replace('\n', '')
        _CATEGORY_ELEM = {
            "supercategory": "pizza_ingredient",
            "id": i,
            "name": c
        }
        _CATEGORIES.append(_CATEGORY_ELEM)

    imglist = sorted(imglist)
    annotlist = sorted(annotlist)
    print(f"Image count : {len(imglist)} | Annotation count : {len(annotlist)}")

    def checking_dataset(_imglist, _annotlist):
        for i, a in zip(_imglist, _annotlist):
            assert os.path.basename(i).split('.')[0] == os.path.basename(a).split('.')[0], f"image file {os.path.basename(i)} hasn't annotation file!"

    checking_dataset(imglist, annotlist)

    _dict = COCO_DEFAULT_FORMAT()
    _dict['categories'] = _CATEGORIES

    _IMAGES, _ANNOTATIONS = [], []

    for index, data in enumerate(zip(imglist, annotlist)):
        image, annotation = data
        print(annotation)

        basename = os.path.basename(image)
        img = Image.open(image)
        width, height = img.size

        _IMAGE_ELEM = {
            "license": 1,
            "file_name": basename,
            "coco_url": "",
            "height": height,
            "width": width,
            "date_captured": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "flickr_url": "",
            "id": index,
        }
        _IMAGES.append(_IMAGE_ELEM)

        annotlines = open(annotation).readlines()
        min_x, min_y, max_x, max_y = np.inf, np.inf, -1 * np.inf, -1 * np.inf

        for line in annotlines:
            line = line.split(" ")
            if len(line) < 6:
                continue
            cat_id = int(line[0])
            del line[0]

            line = list(map(float, line))
            for li, l in enumerate(line):
                # even number
                if li == 0 or li % 2 == 0:
                    l = np.round(l * width, 2)
                    if l < min_x:
                        min_x = l
                    if l > max_x:
                        max_x = l

                # odd number
                elif li != 0 and li % 2 == 1:
                    l = np.round(l * height, 2)
                    if l < min_y:
                        min_y = l
                    if l > max_y:
                        max_y = l

                line[li] = l
            area = int(len(line) / 2)

            _ANNOTATION_ELEM = {
                "id": len(_ANNOTATIONS),
                "image_id": int(index),
                "category_id": int(cat_id),
                "iscrowd": 0,
                "segmentation": [line],
                "area": area,
                "bbox": [min_x, min_y, np.round(float(max_x - min_x), 2), np.round(float(max_y - min_y), 2)]
            }

            _ANNOTATIONS.append(_ANNOTATION_ELEM)

    _dict['images'] = _IMAGES
    _dict['annotations'] = _ANNOTATIONS

    return _dict


if __name__ == "__main__":
    # ======================= READ =======================
    """
    # with open("/home/bulgogi/bolero/dataset/coco_minitrain/instances_minitrain2017.json", 'r') as f:
    with open("/home/bulgogi/bolero/dataset/dsc_dataset/roboflow/Guide_Real_Real_2.v3i.coco-segmentation/train/_annotations.coco.json") as f:
        data = json.load(f)

    read(data)
    """
    # ======================= READ =======================

    # ======================= WRITE =======================
    dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ins_seg_dataset/total/train'
    imglist = glob(os.path.join(dataset_rootpath, "images", "*.jpg"))
    annotlist = glob(os.path.join(dataset_rootpath, "labels", "*.txt"))

    json_dict = make_json_dict(imglist, annotlist, ['dough', 'tomato_sauce', 'mozzarella_cheese', 'pepperoni'], fromtype='yolo')
    with open(os.path.join(dataset_rootpath, f"train_dataset_coco.json"), 'w') as f:
        json.dump(json_dict, f, indent=2)
    # ======================= WRITE =======================
