import json
import os
from glob import glob


if __name__ == "__main__":
    rootpath = 'dataset/total'
    labellist = []

    annotlist = glob(os.path.join(rootpath, 'labels', '*.json'))

    for a in annotlist:
        f = open(a, 'r')
        jsondata = json.load(f)

        new_sentences = []
        newname = a.replace(".json", ".txt")
        print(newname)

        labels = jsondata['labels']
        points = jsondata['bboxes']
        iw, ih = jsondata['img_width'], jsondata['img_height']

        for l in labels:
            if l not in labellist:
                labellist.append(l)
        
        for l, p in zip(labels, points):
            xmin, ymin, xmax, ymax = p

            xmin = int(xmin * iw)
            ymin = int(ymin * ih)
            xmax = int(xmax * iw)
            ymax = int(ymax * ih)

            sentence = f"{l} 0 3 -1 {xmin} {ymin} {xmax} {ymax} -1 -1 -1 -1 -1 -1 -1\n"
            new_sentences.append(sentence)
    
        f = open(newname, 'w')
        f.writelines(new_sentences)
    print("categories :", labellist)

