import os
import sys
import numpy as np
from PIL import Image

from glob import glob
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from commons import get_colormap

COLORMAP = get_colormap(256)
CMAP_LIST = COLORMAP.tolist()
PALETTE = [value for color in CMAP_LIST for value in color]

rootpath = "/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped"
savepath = os.path.join(rootpath, "combined")
os.makedirs(savepath, exist_ok=True)

saucelist = glob(os.path.join(rootpath, "sauce_inference", "*.png"))
cheeselist = glob(os.path.join(rootpath, "cheese_annotations", "*.png"))

for s in saucelist:
    sauce = Image.open(s)
    arr_sauce = np.array(sauce)

    basename = os.path.basename(s)
    cheese_annotname = os.path.join(rootpath, "cheese_annotations", basename)
    assert os.path.isfile(cheese_annotname)

    cheese = Image.open(cheese_annotname)
    arr_cheese = np.array(cheese)

    output = arr_sauce + arr_cheese
    output = np.where(output == 3, 2, output)
    print(np.unique(output))
    
    img_png = Image.fromarray(output).convert('P')
    img_png.putpalette(PALETTE)
    img_png.save(os.path.join(savepath, basename))
