import os
from glob import glob
import cv2


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ingredients/v1/dough'
    imglist = glob(os.path.join(dataset_rootpath, "**", "*.jpg"), recursive=True)

    for i in imglist:
        img = cv2.imread(i, cv2.IMREAD_COLOR)

        ih, iw, ic = img.shape
        new_iw = 416 if abs(416 - iw) < abs(640 - iw) else 640
        new_ih = 416 if abs(416 - ih) < abs(640 - ih) else 640

        img = cv2.resize(img, (new_iw, new_ih))
        print(f"{i} : Convert ({iw}, {ih}) to ({new_iw}, {new_ih})")

        cv2.imwrite(i, img)
