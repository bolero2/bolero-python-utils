import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped'
imagelist = sorted(glob(os.path.join(rootpath, "images/*.jpg")))
viewmode = True

for i, imgfile in tqdm(enumerate(imagelist), total=len(imagelist), desc='Checking shape and pair'):
    annotfile = os.path.join(rootpath, "annotations", os.path.basename(imgfile).replace(".jpg", ".png"))
    assert os.path.isfile(imgfile)
    assert os.path.isfile(annotfile)

    img = cv2.imread(imgfile)
    ih, iw, ic = img.shape

    annot = Image.open(annotfile)
    aw, ah = annot.size

    if aw != iw or ah != ih:
        print("Different filename :", os.path.basename(imgfile))
        exit()

    if viewmode:
        cv2.imshow("image", img)
        cv2.imshow("annotation", np.array(annot))
        key = cv2.waitKey(0)

        if key == ord('d'):
            os.remove(imgfile)
            os.remove(annotfile)
