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
CROP = True
ELASTIC_DEFORM = True


if __name__ == "__main__":
    rootpath = '/home/bulgogi/bolero/dataset/dsc_dataset/total_aug/train'

    savepath = os.path.join(rootpath, "augmented")
    if os.path.isdir(savepath):
        sh.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)

    flip_savepath = os.path.join(savepath, "flip")
    crop_savepath = os.path.join(savepath, "crop")
    elastic_deform_savepath = os.path.join(savepath, "elastic_deform")
    filtered_savepath = os.path.join(savepath, "filtered")

    if os.path.isdir(flip_savepath):
        sh.rmtree(flip_savepath)

    if os.path.isdir(crop_savepath):
        sh.rmtree(crop_savepath)

    if os.path.isdir(elastic_deform_savepath):
        sh.rmtree(elastic_deform_savepath)

    if os.path.isdir(filtered_savepath):
        sh.rmtree(filtered_savepath)

    os.makedirs(flip_savepath, exist_ok=True)
    os.makedirs(crop_savepath, exist_ok=True)
    os.makedirs(elastic_deform_savepath, exist_ok=True)
    os.makedirs(filtered_savepath, exist_ok=True)

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
        if FLIP:            # tag = flip_
            annotfile = imgfile.replace("/images/", "/annotations/").replace(".jpg", ".png")
            annot = Image.open(annotfile)

            flip_ret = aug.flip(img, annot)

            for index, data in enumerate(zip(flip_ret[0], flip_ret[1])):
                f_img, f_annot = data
                cv2.imwrite(os.path.join(flip_savepath, f"flip_{index}_{basename}"), f_img)
                f_annot.save(os.path.join(flip_savepath, f"flip_{index}_{os.path.basename(annotfile)}"))

        if ELASTIC_DEFORM:  # tag = ed_
            annotfile = imgfile.replace("/images/", "/annotations/").replace(".jpg", ".png")
            annot = Image.open(annotfile)

            ed_ret = aug.elastic_deform(img, annot, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)

            ed_img, ed_annot = ed_ret
            cv2.imwrite(os.path.join(elastic_deform_savepath, f"ed_{basename}"), ed_img)
            ed_annot.save(os.path.join(elastic_deform_savepath, f"ed_{os.path.basename(annotfile)}"))

        if CROP:            # tag = crop_
            annotfile = imgfile.replace("/images/", "/annotations/").replace(".jpg", ".png")
            annot = Image.open(annotfile)
            label = list(np.unique(np.array(annot)).tolist())

            if 2 in label and 3 in label:
                crop_ret = aug.crop(img, annot)
                for index, data in enumerate(zip(crop_ret[0], crop_ret[1])):
                    c_img, c_annot = data
                    cv2.imwrite(os.path.join(crop_savepath, f"crop_{index}_{basename}"), c_img)
                    c_annot.save(os.path.join(crop_savepath, f"crop_{index}_{os.path.basename(annotfile)}"))

    imglist = glob(os.path.join(rootpath, "images", "*.jpg")) + glob(os.path.join(rootpath, "augmented", "**", "*.jpg"))

    for index, imgfile in enumerate(tqdm(imglist, total=len(imglist), desc="filtering process")):
        img = cv2.imread(imgfile)
        basename = os.path.basename(imgfile)

        rint = random.randint(0, len(filtering) - 1)

        filtered = eval(filtering[rint])(img) if int(rint) != 0 else img
        filtered = aug.gamma_correction(filtered)

        cv2.imwrite(os.path.join(filtered_savepath, basename), filtered)
