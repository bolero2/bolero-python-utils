import os
from glob import glob
import shutil as sh


path = '/home/bulgogi/bolero/dataset/aimmo/exp2/saved'

annotpath = path
imglist = glob(os.path.join(path, "**", "*.jpg"), recursive=True)

for index, i in enumerate(imglist):
    basename = os.path.basename(i)
    dirname = os.path.dirname(i)

    # annotfile = os.path.join(annotpath, basename.replace('.png', '.txt'))

    os.rename(i, os.path.join(dirname, f"pc_exp2_{index}.jpg"))
    # os.rename(annotfile, os.path.join(dirname, f"snue1_{index}.txt"))
