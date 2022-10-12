import os
import cv2
from glob import glob
from PIL import Image
from tqdm import tqdm


imagelist = sorted(glob("images/*.jpg"))

for i, imgfile in tqdm(enumerate(imagelist), total=len(imagelist), desc='Checking shape and pair'):
    annotfile = os.path.join("annotations", os.path.basename(imgfile).replace(".jpg", ".png"))
    assert os.path.isfile(imgfile)
    assert os.path.isfile(annotfile)

    img = cv2.imread(imgfile)
    ih, iw, ic = img.shape

    annot = Image.open(annotfile)
    aw, ah = annot.size

    if aw != iw or ah != ih:
        print("Different filename :", os.path.basename(imgfile))
        exit()
