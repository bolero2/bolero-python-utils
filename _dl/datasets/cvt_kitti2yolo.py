import json
import os
import base64
import cv2
from glob import glob


CLASSES = ['car', 'bus', 'truck', 'person', 'bicycle', 'motorcycle']

if __name__ == "__main__":
    rootpath = 'dataset/total'
    save_rootpath = 'yolo-labels'

    imgpath = os.path.join(rootpath, 'jpg-images')
    labelpath = os.path.join(rootpath, 'kitti-labels')
    savepath = os.path.join(rootpath, save_rootpath)

    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    imglist = glob(os.path.join(imgpath, '*.jpg'))
    annotlist = glob(os.path.join(labelpath, '*.txt'))


    for imgfile in imglist:
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        yolo_txtfile = os.path.basename(imgfile).replace(".jpg", ".txt")

        ih, iw, ic = img.shape
                    
        annotfile = imgfile.replace('jpg-images', 'kitti-labels').replace('.jpg', '.txt')
        assert annotfile in annotlist, "Wrong Annotation file!"

        f = open(annotfile, 'r')
        lines = f.readlines()
        f.close()

        sentences = []

        for line in lines:
            _line = line.split(' ')
            classname = _line[0]
            bbox = _line[4:8]
            xmin, ymin, xmax, ymax = bbox

            classindex = CLASSES.index(classname)

            yolo_xmin = float(float(xmin) / iw)
            yolo_ymin = float(float(ymin) / ih)
            yolo_xmax = float(float(xmax) / iw)
            yolo_ymax = float(float(ymax) / ih)

            sentence = f"{classindex} {yolo_xmin} {yolo_ymin} {yolo_xmax} {yolo_ymax}\n"
            sentences.append(sentence)

        with open(os.path.join(savepath, yolo_txtfile), 'w') as txt_file:
            txt_file.writelines(sentences)

    CLASSES = [x + "\n" for x in CLASSES]
    print(CLASSES)
    with open(os.path.join(savepath, "classes.txt"), 'w') as f:
        f.writelines(CLASSES)

