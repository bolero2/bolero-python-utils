from glob import glob
from random import shuffle
import os
import shutil as sh

# rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/TRAIN/aimmo_dataset/1/original'
# rootpath = '/home/bulgogi/bolero/dataset/coco_minitrain/yolo/total'
rootpath = '/home/bulgogi/bolero/dataset/aistt_ins_seg_dataset/total'
imglist = glob(os.path.join(rootpath, 'images', '*.jpg'))
shuffle(imglist)

print("Image Count :", len(imglist))
count = len(imglist)

for i in range(count):
    t_image = imglist[i]
    t_annot = os.path.join(rootpath, "annotations", os.path.splitext(os.path.basename(t_image))[0] + ".txt")

    assert os.path.isfile(t_image) and os.path.isfile(t_annot)

    if i < int(count * 0.8):
        savepath = 'train'

    elif int(count * 0.8) <= i < int(count * 0.9):
        savepath = 'valid'

    elif int(count * 0.9) < i:
        savepath = 'test'

    if not os.path.isdir(os.path.join(rootpath, savepath)):
        os.makedirs(os.path.join(rootpath, savepath), exist_ok=True)

    if not os.path.isdir(os.path.join(rootpath, savepath, "images")):
        os.makedirs(os.path.join(rootpath, savepath, "images"), exist_ok=True)
    if not os.path.isdir(os.path.join(rootpath, savepath, "labels")):
        os.makedirs(os.path.join(rootpath, savepath, "labels"), exist_ok=True)

    sh.copy(t_image, os.path.join(rootpath, savepath, "images"))
    sh.copy(t_annot, os.path.join(rootpath, savepath, "labels"))
