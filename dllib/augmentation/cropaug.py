import cv2
from PIL import Image
import numpy as np


def geometric_crop(img, annot, rate=0.7):
    ih, iw, ic = img.shape
    
    img_crop1 = img[0:int(ih * rate), 0:int(iw * rate), :]
    annot_crop1 = annot.crop((0, 0, int(iw * rate), int(ih * rate)))

    img_crop2 = img[int(ih * (1 - rate)):ih, 0:int(iw * rate), :]
    annot_crop2 = annot.crop((0, int(ih * (1 - rate)), int(iw * rate), ih))

    img_crop3 = img[int(ih * (1 - rate)):ih, int(iw * (1 - rate)):iw, :]
    annot_crop3 = annot.crop((int(iw * (1 - rate)), int(ih * (1 - rate)), iw, ih))

    img_crop4 = img[0:int(ih * rate), int(iw * (1 - rate)):iw, :]
    annot_crop4 = annot.crop((int(iw * (1 - rate)), 0, iw, int(ih * rate)))

    img_crop5 = img[int(ih * (1 - rate)):int(ih * rate), int(iw * (1 - rate)):int(iw * rate), :]
    annot_crop5 = annot.crop((int(iw * (1 - rate)), int(ih * (1 - rate)), int(iw * rate), int(ih * rate)))

    cropped_list = [[img_crop1, img_crop2, img_crop3, img_crop4, img_crop5], [annot_crop1, annot_crop2, annot_crop3, annot_crop4, annot_crop5]]

    return cropped_list


if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    annot = Image.open("sample.png")

    ret = geometric_crop(img, annot)

    for i, data in enumerate(zip(ret[0], ret[1])):
        _img, _annot = data

        cv2.imwrite(f"crop_{i}_sample.jpg", _img)
        _annot.save(f"crop_{i}_sample.png")
