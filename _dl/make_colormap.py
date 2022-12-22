import os
import sys
import shutil as sh
from glob import glob
import numpy as np
import json
from PIL import Image, ImageDraw
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

from bcommon import get_colormap


rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dough_data'
imagelist = glob(os.path.join(rootpath, "images", "*.jpg"))

cmap = get_colormap(256).tolist()
palette = [value for color in cmap for value in color]
class_info = [x.replace("\n", '') for x in open(os.path.join(rootpath, "class.txt"), 'r').readlines()]

for i in imagelist:
    is_dough = False
    basename = os.path.basename(i)

    jsonfile = i.replace(".jpg", ".json").replace("/images/", "/jsons/")
    savepath = os.path.join(rootpath, "annotations")

    if os.path.isfile(jsonfile):
        with open(jsonfile, 'r') as f:
            jsondata = json.load(f)

    else:
        continue

    ih, iw = int(jsondata['imageHeight']), int(jsondata['imageWidth'])
    print(" ->", i, f" | {iw} + {ih} is saved in {savepath}")

    img_png = np.zeros((ih, iw), np.uint8)
    img_png = Image.fromarray(img_png)
    draw = ImageDraw.Draw(img_png)

    labels = [x['label'] for x in jsondata['shapes']]
    if 'dough' not in labels:
        continue

    else:
        # 1. only dough
        for elem in jsondata['shapes']:
            label = elem['label']
            if label == 'dough':
                label_index = class_info.index(label)
                points = elem['points']
                points = [tuple(x) for x in points]

                draw.polygon(points, fill=int(label_index))

        # img_png = Image.fromarray(img_png).convert('P')
        img_png = img_png.convert('P')
        img_png.putpalette(palette)
        img_png.save(os.path.join(savepath, basename.replace(".jpg", ".png")))
        # sh.copy(i, os.path.join(savepath, "images"))
