from glob import glob
import os
from PIL import Image
import numpy as np
import shutil as sh
from tqdm import tqdm
import sys
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from bcommon import get_colormap

COLORMAP = get_colormap(256)
CMAP_LIST = COLORMAP.tolist()
PALETTE = [value for color in CMAP_LIST for value in color]


path1 = "dough_result"
path2 = 'total/annotations+2_3'

list1 = sorted(glob(os.path.join(path1, "*.png")))
list2 = sorted(glob(os.path.join(path2, "*.png")))
savepath = 'annotations+1_2_3'

if os.path.isdir(savepath):
    sh.rmtree(savepath)
os.makedirs(savepath, exist_ok=True)

for index, elem in tqdm(enumerate(zip(list1, list2)), total=len(list1)):
    print(elem)
    img1_filename, img2_filename = elem

    img1 = np.array(Image.open(img1_filename))
    img2 = np.array(Image.open(img2_filename))

    assert img1.shape == img2.shape, "Have different shape!"
    basename = os.path.basename(img1_filename)
    ih, iw = img1.shape

    canvas = np.zeros_like(img1).astype(np.uint8)

    for h in range(ih):
        for w in range(iw):
            val1 = int(img1[h, w])
            val2 = int(img2[h, w])
            val = val1 + val2
            if val1 != 0 and val2 != 0:
                val = val2

            canvas[h, w] = val

    img_png = Image.fromarray(canvas).convert('P')
    img_png.putpalette(PALETTE)
    img_png.save(os.path.join(savepath, basename))
