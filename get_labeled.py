from glob import glob
import sys
import os
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
import shutil as sh
import cv2
from util import convert_coordinate

root_path = '/Users/bolero/dc/dataset/WiderFaceDetectionDataset'
annotlist = glob(os.path.join(root_path, "wider_face_split", "wider_face_*_bbx_gt.txt"))

for annotfile in annotlist:
    image_root_path = os.path.join(root_path, "WIDER_val" if "val" in annotfile else "WIDER_train")

    f = open(annotfile, 'r')
    lines = f.readlines()
    name = ""
    imgpath = ""

    sentences = []

    for line in lines:
        line = line.replace("\n", "")

        if 'jpg' in line:
            if len(sentences) != 0:
                annotname = os.path.join(image_root_path, "annotations", os.path.basename(imgpath).replace("jpg", "txt"))
                print(" --> save annotation text name :", annotname, " | saved annotations count :", len(sentences))
                f2 = open(annotname, 'w')
                f2.writelines(sentences)
                f2.close()

                sentences = []
            else:
                pass
            name = line
            imgpath = os.path.join(image_root_path, "images", line)
            print("Image path :", os.path.basename(imgpath))
            assert os.path.isfile(imgpath)
            img = cv2.imread(imgpath)
            height, width, ch = img.shape

        elif 'jpg' not in line and len(line.split(' ')) < 4:
            pass
        elif 'jpg' not in line and len(line.split(' ')) > 4:
            xywh = line.split(' ')[0:2] + line.split(' ')[2:4]
            xywh = list(map(int, xywh))
            get_ccwh = convert_coordinate(xywh, (height, width), "xywh", "abs", "ccwh", "relat")
            ccwh = get_ccwh()
            
            new_sentence = f"0 {ccwh[0]} {ccwh[1]} {ccwh[2]} {ccwh[3]}\n"
            sentences.append(new_sentence)
    
    if len(sentences) != 0:
        annotname = os.path.join(image_root_path, "annotations", os.path.basename(imgpath).replace("jpg", "txt"))
        print(" --> save annotation text name :", annotname, " | saved annotations count :", len(sentences))
        f2 = open(annotname, 'w')
        f2.writelines(sentences)
        f2.close()

        sentences = []
