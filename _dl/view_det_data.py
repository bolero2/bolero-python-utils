import os
from glob import glob
import numpy as np
import shutil as sh
import sys
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from commons import draw_bbox_image
import cv2


# path = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/total'
path = '/home/bulgogi/bolero/dataset/det_dough/total/images'
savepath = ''
savepath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/dough/'
imageext = 'jpg'
annotext = 'txt'

imagelist = glob(os.path.join(path, f"**/*.{imageext}"), recursive=True)
if savepath != '':
    os.makedirs(savepath, exist_ok=True) 

for i in imagelist:
    annotfile = i.replace(f".{imageext}", f".{annotext}").replace("images", "annotations")
    if not os.path.isfile(annotfile):
        continue

    annotlines = open(annotfile, 'r').readlines()

    annotlines = [x.split(' ') for x in annotlines]
    arr = []
    if annotlines == []:
        continue
    for a in annotlines:
        label = int(a[0])
        coord = list(map(float, a[1:]))
        arr.append([label, *coord])
    arr = np.array(arr)
    arr = arr[:, 1:].tolist()

    img = draw_bbox_image(i, arr)
    cv2.imshow("img", img)
    if cv2.waitKey(0) == ord('q'):
        break

    elif cv2.waitKey(0) == ord('s') and savepath != '':
        sh.copy(i, savepath)
        sh.copy(annotfile, savepath)
        print("saved.")
