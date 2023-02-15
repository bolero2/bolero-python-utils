import os
import cv2
import shutil as sh
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    rootpath = "/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/TRAIN/aimmo_dataset/1/total/train"
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
        annot_np = np.array(annot)
        if np.unique(annot_np).tolist() == [0]:
            print("Only zero label :", os.path.basename(annotfile))

        if aw != iw or ah != ih:
            print("Different filename :", os.path.basename(imgfile))
            exit()


    annotlist = sorted(glob(os.path.join(rootpath, "annotations", "*.png")))

    for i, annotfile in tqdm(enumerate(annotlist), total=len(annotlist), desc="Checking remained annotation file"):
        imgfile = os.path.join(rootpath, "images", os.path.basename(annotfile).replace(".png", ".jpg"))
        assert os.path.isfile(annotfile)

        if not os.path.isfile(imgfile):
            sh.move(annotfile, trashbox)
            continue
