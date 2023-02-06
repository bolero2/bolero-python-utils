from flipaug import filtering1, filtering2
import random
import cv2
from glob import glob
import os


rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dough_data/new_total_dataset/images'
imglist = glob(os.path.join(rootpath, "*.png"), recursive=True)

for imgname in imglist:
    img = cv2.imread(imgname, cv2.IMREAD_COLOR)

    new_img = eval(f"filtering{random.randint(1, 2)}")(img)
    cv2.imwrite(imgname, new_img)
