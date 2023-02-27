import cv2
import numpy as np
from glob import glob
import os


def grayscaling(img:np.array):
    canvas = np.zeros_like(img)
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canvas[:, :, 0] = dst
    canvas[:, :, 1] = dst
    canvas[:, :, 2] = dst

    return canvas 


if __name__ == "__main__":
    rootpath = '/home/bulgogi/bolero/dataset/dsc_dataset/only_dsc/grayscaled/test/original_images/'
    imglist = glob(os.path.join(rootpath, "*.jpg"), recursive=True)
    print(len(imglist))

    for idx, imgname in enumerate(imglist):
        basename = os.path.basename(imgname)
        img = cv2.imread(imgname, cv2.IMREAD_COLOR)
        new_img = grayscaling(img)

        cv2.imwrite(imgname, new_img)
