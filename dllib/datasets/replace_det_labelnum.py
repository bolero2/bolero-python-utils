import os
from glob import glob


path = '/home/bulgogi/bolero/dataset/dsc_dataset/original/instance_segmentation/total/pepperoni_det_result/final/labels/'
savepath = path

annotlist = glob(os.path.join(path, "*.txt"))

for a in annotlist:
    lines = open(a, 'r').readlines()

    new_sentence = []

    for l in lines:
        l = l.split(' ')
        if l[0] != '3':
            l[0] = '3'

        new_sentence.append(' '.join(l))

    with open(os.path.join(savepath, os.path.basename(a)), 'w') as f:
        f.writelines(new_sentence)
