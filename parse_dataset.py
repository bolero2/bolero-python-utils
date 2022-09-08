from glob import glob
import os
from tqdm import tqdm
import cv2


def get_dataset(imgpath, annotpath):
    """
    Input: image path, annotation path(txt)
    Return: [image list, annotation list, width, height]

        imagelist -> abs path
        annotation list -> ccwh type, [label, center_x, center_y, width, height]
        widths -> group of image width 
        heights -> group of image height
    """

    datalist = []
    images, annotations = [], []
    widths, heights = [], []

    imglist = glob(os.path.join(imgpath, "*.jpg"))

    for i, imgname in tqdm(enumerate(imglist), total=len(imglist), desc='Parsing Dataset ... '):
        images.append(imgname)

        img = cv2.imread(imgname)
        ih, iw, ic = img.shape

        widths.append(iw)
        heights.append(ih)

        basename = os.path.basename(imgname)
        annotname = os.path.join(annotpath, basename.replace("jpg", "txt"))

        annotfile = open(annotname, "r")
        annotlines = annotfile.readlines()
        temp_annots = []

        for a in annotlines:
            annotval = list(map(float, a.split(' ')))
            annotval[0] = int(annotval[0])
            temp_annots.append(annotval)
        annotations.append(temp_annots)

        datalist = [images, annotations, widths, heights]
    
    return datalist
    

if __name__ == "__main__":
    res = get_dataset("/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/images", "/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/annotations") 

    print(res)
