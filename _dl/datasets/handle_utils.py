import os
import shutil as sh
import numpy as np
import sys
import json
from datetime import datetime
import cv2
from PIL import Image

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

from commons import convert_coordinate as cc


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


def read(data, imgfile, dtype='labelme', classes=''):
    if dtype == 'labelme':
        categories = []
        for i in range(len(data['shapes'])):
            category_name = data['shapes'][i]['label']
            categories.append(category_name)

        for i, c in enumerate(categories):
            print(f"  {i}: {c}")

        polygons, polygon = [], []

        for index in range(len(data['shapes'])):
            for elem in data['shapes'][index]['points']:
                polygon += elem
            polygons.append(polygon)
            polygon = []

        _ret = {
            "width": data['imageWidth'],
            "height": data['imageHeight'], 
            "polygon": polygons,
            "category": categories,
            "filename": data['imagePath'] }

        return _ret

    elif dtype == 'yolo':
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        ih, iw, ic = img.shape

        _dict = {
            'version': '5.2.1',
            'flags': {},
            'shapes': [],
            'imagePath': os.path.basename(imgfile),
            'imageData': None,
            'imageHeight': ih,
            'imageWidth': iw
        }

        for d in data:
            elem = {}

            label_index = int(d.split(' ')[0])
            single_cls = True if isinstance(classes, str) else False
            if single_cls:
                label_name = classes
            else:
                label_name = classes[label_index]

            polygons = list(map(float, d.split(' ')[1:]))
            polygons[0::2] = np.array(polygons[0::2]) * iw 
            polygons[1::2] = np.array(polygons[1::2]) * ih
            _polygons = list(map(lambda x:np.round(x, 1), polygons))
            polygons = _polygons
            del _polygons
            points = []

            for i in range(0, len(polygons), 2):
                points.append(polygons[i:i + 2])

            elem['label'] = label_name
            elem['group_id'] = None
            elem['shape_type'] = 'polygon'
            elem['flags'] = {}
            elem['points'] = points

            _dict['shapes'].append(elem)

        return _dict
    else:
        raise NotImplementedError


def labelme2yolo(data, categories, dataset_rootpath='', savedir='', savetype='total'):
    width, height = data['width'], data['height']
    filename = data['filename']
    labels = data['category']
    points = data['polygon']

    if savetype == 'total':
        savedir = os.path.join(dataset_rootpath, savetype, savedir)
        sentences = []
        for l, p in zip(labels, points):
            l = l.replace("\\", "").lower()
            if l == "mozzarella":
                l = "mozzarella_cheese"
            elif l == 'sauce':
                l = 'tomato_sauce'
            # label_index = categories.index(l)
            label_index = categories.index(l)

            p[0::2] = np.array(p[0::2]) / width
            p[1::2] = np.array(p[1::2]) / height

            p = list(map(str, p))
            points_sentence = ' '.join(p)
            sentences.append(f"{label_index} {points_sentence}\n")

        if savedir != '' and not os.path.isdir(savedir):
            print("  Making save directory :", savedir)
            os.makedirs(savedir, exist_ok=True)    

        savepath = os.path.join(savedir, filename.replace(".jpg", ".txt"))
        print("\t saved in", savepath)
        with open(savepath, 'w') as f:
            f.writelines(sentences)

        print()

    elif savetype == 'ingredients':
        savedir = os.path.join(dataset_rootpath, savetype)
        save_names = {}
        sentences = []
        for l, p in zip(labels, points):        # l = label_name
            label_index = 0
            # print(p)
            p[0::2] = np.array(p[0::2]) / width
            p[1::2] = np.array(p[1::2]) / height

            p = list(map(str, p))
            points_sentence = ' '.join(p)
            points_sentence = f"0 {points_sentence}\n"

            temp_sentences_list = save_names.get(l, [])
            temp_sentences_list.append(points_sentence)
            save_names[l] = temp_sentences_list

        for target_name, target_points in save_names.items():
            save_image_path = os.path.join(savedir, target_name, "images")
            save_label_path = os.path.join(savedir, target_name, "labels")

            if not os.path.isdir(save_image_path):
                os.makedirs(save_image_path, exist_ok=True)
            if not os.path.isdir(save_label_path):
                os.makedirs(save_label_path, exist_ok=True)

            sh.copy(os.path.join(dataset_rootpath, "images", filename), save_image_path)
            with open(os.path.join(save_label_path, filename.replace(".jpg", ".txt")), 'w') as f:
                print("\t saved in", os.path.join(save_label_path, filename.replace(".jpg", ".txt")))
                f.writelines(save_names[target_name])

        print()
