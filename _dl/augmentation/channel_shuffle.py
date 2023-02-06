import numpy as np
import random
from itertools import permutations


def channel_shuffle(img:np.array):
    if len(img.shape) == 2 or img.shape[-1] == 1:
        return img

    dst = np.zeros_like(img) 
    channels = [x for x in range(dst.shape[-1])]
    pmt = list(permutations(channels))

    randint = random.randint(0, len(pmt) - 1)

    for i, ch in enumerate(pmt[randint]):
        dst[:, :, i] = img[:, :, ch]

    return dst


if __name__ == "__main__":
    import cv2

    img = cv2.imread("/home/bulgogi/Desktop/quantiza_test1.jpg")
    ret = channel_shuffle(img)

    cv2.imshow("aa", img)
    cv2.imshow("dst", ret)
    cv2.waitKey(0)
