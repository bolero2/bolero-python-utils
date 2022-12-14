from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import random
from copy import deepcopy
from matplotlib import pyplot as plt


rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped'
imagelist = sorted(glob(os.path.join(rootpath, "images", "*.jpg")))
annotlist = sorted(glob(os.path.join(rootpath, "annotations", "*.png")))
os.makedirs(os.path.join(rootpath, "augmented", "normal", "images"), exist_ok=True)
os.makedirs(os.path.join(rootpath, "augmented", "normal", "annotations"), exist_ok=True)


def filtering2(img):       # Gaussian Blur
    sigma = 7
    dst = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return dst

def filtering1(img):     # Salt and Pepper
    ih, iw, ic = img.shape

    dst = deepcopy(img)

    percentage = random.randint(int(ih * iw / 100), int(ih * iw / 10))

    for i in range(percentage):
        yco = random.randint(0, ih - 1)
        xco = random.randint(0, iw - 1)
        dst[yco, xco, :] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    return dst

for i, data in tqdm(enumerate(zip(imagelist, annotlist)), total=len(imagelist), desc='Flip Augmentation'):
    imgfile = data[0]
    annotfile = data[1]
    assert os.path.basename(imgfile).split('.')[0] == os.path.basename(annotfile).split('.')[0]
    
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    ant = Image.open(annotfile)
    
    # 1. Flip -> (1, -1)
    # print(eval("filtering2"))
    altimg1 = eval(f"filtering{random.randint(1, 2)}")(cv2.flip(img, 1))
    altimg2 = eval(f"filtering{random.randint(1, 2)}")(cv2.flip(img, -1))
    altimg3 = eval(f"filtering{random.randint(1, 2)}")(cv2.flip(img, 0))
    
    altant1 = ant.transpose(Image.FLIP_LEFT_RIGHT)
    altant2 = ant.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
    altant3 = ant.transpose(Image.FLIP_TOP_BOTTOM)
    
    cv2.imwrite(os.path.join(rootpath, "augmented", "normal", "images", f"flip_0_{os.path.basename(imgfile)}"), altimg1)
    cv2.imwrite(os.path.join(rootpath, "augmented", "normal", "images", f"flip_1_{os.path.basename(imgfile)}"), altimg2)
    cv2.imwrite(os.path.join(rootpath, "augmented", "normal", "images", f"flip_2_{os.path.basename(imgfile)}"), altimg3)
    
    altant1.save(os.path.join(rootpath, "augmented", "normal", "annotations", f"flip_0_{os.path.basename(annotfile)}"))
    altant2.save(os.path.join(rootpath, "augmented", "normal", "annotations", f"flip_1_{os.path.basename(annotfile)}"))
    altant3.save(os.path.join(rootpath, "augmented", "normal", "annotations", f"flip_2_{os.path.basename(annotfile)}"))
