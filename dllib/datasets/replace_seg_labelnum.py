import os
import numpy as np
from PIL import Image
from glob import glob
import sys

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from commons import get_colormap

COLORMAP = get_colormap(256)
CMAP_LIST = COLORMAP.tolist()
PALETTE = [value for color in CMAP_LIST for value in color]

rootpath = "/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/TRAIN/segmentation/total/annotations+1_2"
# rootpath = "/home/bulgogi/bolero/dataset/aistt_dataset/dough_data/cropped/annotations"
oldlabel = 3 
newlabel = 2

annotlist = glob(os.path.join(rootpath, "*.png"))

for a in annotlist:
    annot = Image.open(a)
    arr_annot = np.array(annot)

    arr_annot = np.where(arr_annot == oldlabel, newlabel, arr_annot)

    img_png = Image.fromarray(arr_annot).convert('P')
    img_png.putpalette(PALETTE)
    img_png.save(a)
