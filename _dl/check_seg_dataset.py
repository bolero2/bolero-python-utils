import os
import cv2
import shutil as sh
from glob import glob
from PIL import Image
from tqdm import tqdm


rootpath = "/home/bulgogi/bolero/dataset/aistt_dataset/dough_data/cropped"
imagelist = sorted(glob(os.path.join(rootpath, "images", "*.jpg")))
trashbox = os.path.join(rootpath, "trash")
os.makedirs(trashbox, exist_ok=True)

for i, imgfile in tqdm(enumerate(imagelist), total=len(imagelist), desc='Checking shape and pair'):
    annotfile = os.path.join(rootpath, "annotations", os.path.basename(imgfile).replace(".jpg", ".png"))
    assert os.path.isfile(imgfile)

    if not os.path.isfile(annotfile):# , f"{annotfile} is None."
        sh.move(imgfile, trashbox)
        continue

    img = cv2.imread(imgfile)
    ih, iw, ic = img.shape

    annot = Image.open(annotfile)
    aw, ah = annot.size

    if aw != iw or ah != ih:
        print("Different filename :", os.path.basename(imgfile))
        exit()
