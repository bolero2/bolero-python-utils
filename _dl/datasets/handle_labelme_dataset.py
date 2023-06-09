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


def read(json_data, dtype='labelme'):
    if dtype == 'labelme':
        # print("\n")
        categories = []
        # print(data.keys())
        # print(data['version'])
        # print(data['flags'])
        for i in range(len(data['shapes'])):
            category_name = data['shapes'][i]['label']
            categories.append(category_name)
            # print(data['categories'][i]['name'])

        # print(categories)
        for i, c in enumerate(categories):
            print(f"  {i}: {c}")

        # print(f"  Image Width : {data['imageWidth']} | Image Height : {data['imageHeight']}")
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
        return None
    else:
        raise NotImplementedError


def convert_dataset(data, categories, dest='yolo', dataset_rootpath='', savedir='', savetype='total'):
    width, height = data['width'], data['height']
    filename = data['filename']
    labels = data['category']
    points = data['polygon']

    if savetype == 'total':
        savedir = os.path.join(dataset_rootpath, savetype, savedir)
        sentences = []
        for l, p in zip(labels, points):
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

        with open(os.path.join(savedir, filename.replace(".jpg", ".txt")), 'w') as f:
            f.writelines(sentences)

    elif savetype == 'ingredients':
        savedir = os.path.join(dataset_rootpath, savetype)
        save_names = {}
        sentences = []
        for l, p in zip(labels, points):        # l = label_name
            label_index = 0
            print(p)
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
                f.writelines(save_names[target_name])


if __name__ == "__main__":
    # dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ingredients/dough+tomato_sauce/'
    # dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ingredients/dough'
    dataset_rootpath = '/home/bulgogi/Desktop/margeritta_jsons/'
    categories=['dough', 'tomato_sauce', 'mozzarella_cheese', 'pepperoni', 'basil_oil', 'marinated_tomato', 'meat_sauce'] 

    jsonlist = glob(os.path.join(dataset_rootpath, "jsons", "*.json"))

    annotations = []

    # ======================= READ =======================
    for jsonfile in jsonlist:
        print("Target file :", jsonfile)
        with open(jsonfile) as f:
            data = json.load(f)
        _annotation = read(data, dtype='labelme')
        annotations.append(_annotation)
    # ======================= READ =======================

    # ======================= WRITE =======================
    for annot in annotations:
        convert_dataset(annot, savedir='labels', dataset_rootpath=dataset_rootpath, categories=categories, savetype='total')
        # convert_dataset(annot, dataset_rootpath=dataset_rootpath, categories=categories, savetype='ingredients')
    # ======================= WRITE =======================
