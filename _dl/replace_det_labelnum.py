import os
from glob import glob


path = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/total/annotations'
savepath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/new_annotation'
annotlist = glob(os.path.join(path, "*.txt"))

for a in annotlist:
    lines = open(a, 'r').readlines()

    new_sentence = []

    for l in lines:
        l = l.split(' ')
        if l[0] != '0':
            l[0] = '1'

        new_sentence.append(' '.join(l))

    with open(os.path.join(savepath, os.path.basename(a)), 'w') as f:
        f.writelines(new_sentence)
