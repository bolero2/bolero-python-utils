import cv2
from copy import deepcopy
import random


def salt_and_pepper(img):
    ih, iw, ic = img.shape
    dst = deepcopy(img)
    percentage = random.randint(int(ih * iw / 100), int(ih * iw / 10))
   
    for i in range(percentage):
        xco = random.randint(0, iw - 1)
        yco = random.randint(0, ih - 1)
        dst[yco, xco, :] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    return dst
