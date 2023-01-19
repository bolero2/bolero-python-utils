from glob import glob
from random import shuffle
import os
import shutil as sh

rootpath = 'total'
imglist = glob(os.path.join(rootpath, '*.jpg'))
shuffle(imglist)

print(imglist)
count = len(imglist)

for i in range(count):
    t_image = imglist[i]
    t_annot = os.path.join(rootpath, os.path.splitext(os.path.basename(t_image))[0] + ".png")

    assert os.path.isfile(t_image) and os.path.isfile(t_annot)

    if i < int(count * 0.8):
        savepath = 'train'

    elif int(count * 0.8) <= i < int(count * 0.9):
        savepath = 'valid'

    elif int(count * 0.9) < i:
        savepath = 'test'

    sh.copy(t_image, savepath)
    sh.copy(t_annot, savepath)
