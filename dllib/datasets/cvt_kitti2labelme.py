import json
import os
import base64
import cv2
from glob import glob


if __name__ == "__main__":
    rootpath = 'dataset/total'
    save_rootpath = 'labelme-labels'

    imgpath = os.path.join(rootpath, 'jpg-images')
    labelpath = os.path.join(rootpath, 'kitti-labels')
    savepath = os.path.join(rootpath, save_rootpath)

    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    imglist = glob(os.path.join(imgpath, '*.jpg'))
    annotlist = glob(os.path.join(labelpath, '*.txt'))

    for imgfile in imglist:
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        labelme_jsonfile = os.path.basename(imgfile).replace(".jpg", ".json")

        ih, iw, ic = img.shape
        imgData = str(base64.b64encode(open(imgfile, "rb").read()))

        base_dict = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],   # input points data
            "imagePath": os.path.basename(imgfile),
            "imageData": None,
            "imageWidth": iw,
            "imageHeight": ih
        }
                    
        annotfile = imgfile.replace('jpg-images', save_rootpath).replace('.jpg', '.txt')
        assert annotfile in annotlist, "Wrong Annotation file!"

        f = open(annotfile, 'r')
        lines = f.readlines()
        f.close()

        for line in lines:
            _line = line.split(' ')
            classname = _line[0]
            bbox = _line[4:8]
            xmin, ymin, xmax, ymax = bbox

            element = {
                "label": classname,
                "points": [
                    [
                        float(xmin),
                        float(ymin)
                    ],
                    [
                        float(xmax),
                        float(ymax)
                    ]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }

            base_dict['shapes'].append(element)

        with open(os.path.join(savepath, labelme_jsonfile), 'w') as json_file:
            json.dump(base_dict, json_file, indent=2)

