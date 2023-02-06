import cv2
import numpy as np


def gaussian_blur(img:np.array):
    sigma = 7
    dst = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return dst
