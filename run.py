from commons import image_blending
from glob import glob
import os
import cv2
import numpy as np


if __name__ == "__main__":
    imglist = glob("/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/TRAIN/segmentation/total/images/*.jpg")

    for i in imglist:
        img1 = cv2.imread(i)
        img2 = cv2.imread(i.replace("/images/", "/annotations+1_2_3/").replace(".jpg", ".png"))

        img1 = img1 / 255.
        img2 = img2 / 255.

        new_img = image_blending(img1, img2) * 255
        new_img = new_img.astype(np.uint8)

        cv2.imshow("test", new_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
