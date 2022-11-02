import os
from glob import glob
import numpy as np
import shutil as sh
import sys
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from bcommon import draw_bbox_image
import cv2


path = '/home/bulgogi/bolero/dataset/aistt_dataset/cheese_spread_data/cheese_20_result'
savepath = ''
savepath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data'
imageext = 'jpg'

imagelist = glob(os.path.join(path, f"**/*_cheese_seg.{imageext}"), recursive=True)
if savepath != '':
    os.makedirs(savepath, exist_ok=True) 

for index, i in enumerate(imagelist):
    print('\n')
    basename = os.path.basename(i)
    basename, ext = os.path.splitext(basename)

    original_name = basename.replace("_cheese_seg", "")
    if os.path.isfile(os.path.join(savepath, original_name + ".jpg")):
        continue

    print(f" -> {index + 1}/{len(imagelist)} ({np.round((index / len(imagelist)), 4) * 100}%) {original_name}", end='')

    img = cv2.imread(i, cv2.IMREAD_COLOR)

    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    elif key == ord('f') and savepath != '' or key == ord('s') and savepath != '':
        sh.copy(i, savepath)
        sh.copy(os.path.join(path, f"{original_name}.jpg"), savepath)
        sh.copy(os.path.join(path, f"{original_name}_cheese_mask.png"), savepath)
        sh.copy(os.path.join(path, f"{original_name}_cheese_info.json"), savepath)
        print(" ----------> saved.")
