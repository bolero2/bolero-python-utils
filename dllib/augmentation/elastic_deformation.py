import cv2
from PIL import Image
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
from skimage import color
from skimage import io
import os
import sys
import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from _dl import RGB_to_colormap


colors = [[0, 0, 0], [255, 47, 0], [0, 255, 188], [24, 0, 255]]     # background, dough, cheese, sauce
palette = [value for color in colors for value in color]


def colormap_to_RGB(colormap, shape):
    canvas = np.zeros(shape)
    ih, iw, _ = canvas.shape
    unique = list(np.unique(np.array(colormap)).tolist())

    for h in range(ih):
        for w in range(iw):
            val = np.array(colormap)[h, w]
            canvas[h, w, :] = np.array(colors[val])

    canvas = canvas.astype(np.uint8)

    return canvas


def geometric_elastic_deformation(img:np.array, annot, alpha, sigma, alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    shape_size = shape[:2]

    # annot = colormap_to_RGB(annot, img.shape)

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_TRANSPARENT)

    np_eye = np.eye(3)
    np_eye[:2, :] = M
    coeff = np.linalg.inv(np_eye).flatten()[:6]
    annot = annot.transform(annot.size, Image.AFFINE, coeff, resample=Image.BILINEAR)

    """
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    print(indices)

    img = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
    # annot = map_coordinates(np.array(annot), indices, order=1, mode='reflect').reshape(shape[:2])
    """

    return img, annot


if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    annot = Image.open("sample.png")

    img_t, annot_t = geometric_elastic_deformation(img, annot, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
    cv2.imwrite("sample2.jpg", img_t)
    # annot_t = RGB_to_colormap(annot_t)
    annot_t.save("sample2.png")

    output1 = np.concatenate((img, img_t))

    cv2.imshow("aa", output1)
    # cv2.imshow("BB", annot_t)
    cv2.waitKey(0)
