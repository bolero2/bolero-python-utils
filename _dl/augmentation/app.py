import os
import sys
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)

import cv2
import random
import shutil as sh
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import _dl.augmentation as aug


FLIP = True


if __name__ == "__main__":
    rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/TRAIN/aimmo_dataset/1/total/train'

    savepath = os.path.join(rootpath, "augmented")
    if os.path.isdir(savepath):
        sh.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)

    flip_savepath = os.path.join(savepath, "flip")
    if os.path.isdir(flip_savepath):
        sh.rmtree(flip_savepath)
    os.makedirs(flip_savepath, exist_ok=True)

    imglist = glob(os.path.join(rootpath, "images", "*.jpg"))

    filtering = ['none', 'aug.gaussian_blur', 'aug.salt_and_pepper'] 

    for index, imgfile in enumerate(tqdm(imglist, total=len(imglist), desc='Augmentation process')):
        img = cv2.imread(imgfile)
        basename = os.path.basename(imgfile)

        rint = random.randint(0, len(filtering) - 1)

        filtered = eval(filtering[rint])(img) if int(rint) != 0 else img
        filtered = aug.gamma_correction(filtered)

        """
        cv2.imshow("aa", filtered)
        key = cv2.waitKey(300)
        if key == ord('q'):
            break
        """

        cv2.imwrite(os.path.join(savepath, basename), filtered)
        sh.copy(os.path.join(rootpath, "annotations", basename.replace(".jpg", ".png")), savepath)

        if FLIP:

            annotfile = imgfile.replace("/images/", "/annotations/").replace(".jpg", ".png")
            annot = Image.open(annotfile)

            flip_ret = aug.flip(img, annot)

            rint = random.randint(0, len(filtering) - 1)
            for index, data in enumerate(zip(flip_ret[0], flip_ret[1])):
                img, annot = data
                cv2.imshow("img", img)
                cv2.imshow("annot", np.array(annot))
                cv2.waitKey(100)
                filtered = eval(filtering[rint])(img) if int(rint) != 0 else img

                cv2.imwrite(os.path.join(flip_savepath, f"flip_{index}_{basename}"), img)
                annot.save(os.path.join(flip_savepath, f"flip_{index}_{os.path.basename(annotfile)}"))
