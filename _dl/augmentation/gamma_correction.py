import cv2
import os
import random
import numpy as np
from glob import glob


def gamma_correction(img:np.array, gamma:float):
    img = img.astype(np.float32) / 255.

    new_img = img ** (1 / gamma)
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)

    return new_img


if __name__ == "__main__":
    rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dough_data/cropped/images'
    imglist = glob(os.path.join(rootpath, "*.jpg"), recursive=True)
    print(imglist)

    for idx, imgname in enumerate(imglist):
        gamma = random.uniform(0.3, 2.5)
        print(f"[{idx}] Gamma : {gamma}")

        basename = os.path.basename(imgname)
        img = cv2.imread(imgname, cv2.IMREAD_COLOR)
        new_img = gamma_correction(img, gamma)

        cv2.imwrite(imgname, new_img)

        """
        cv2.imshow("new", new_img)
        cv2.imshow("original", img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        """


"""
    img = cv2.imread("sample.jpg")
gamma = 0.4

img = img.astype(np.float32) / 255.

new_img = img ** (1 / gamma)
new_img = new_img * 255
new_img = new_img.astype(np.uint8)

cv2.imshow("original", img)
cv2.imshow("new", new_img)
cv2.waitKey(0)
"""
