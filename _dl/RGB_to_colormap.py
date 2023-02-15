import os
import cv2 
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


colors = [[0, 0, 0], [255, 47, 0], [0, 255, 188], [24, 0, 255]]     # background, dough, cheese, sauce
palette = [value for color in colors for value in color]

def RGB_to_colormap(img):
    print(np.unique(img))
    ih, iw, _ = img.shape
    canvas = np.zeros((ih, iw))

    for h in range(ih):
        for w in range(iw):
            val = list(img[h, w, :].tolist())
            index = colors.index(val)

            canvas[h, w] = index

    canvas = canvas.astype(np.uint8)
    img_png = Image.fromarray(canvas)
    img_png = img_png.convert('P')
    img_png.putpalette(palette)

    return img_png
