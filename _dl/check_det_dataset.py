import os
from glob import glob
from random import shuffle
import shutil as sh


rootpath = '/home/bulgogi/hdisk/'
ext = 'png'

imagelist = glob(os.path.join(rootpath, "snue_folder1", "folder1", f"*.{ext}"))

shuffle(imagelist)
os.makedirs(os.path.join(rootpath, "non_annotations"), exist_ok=True)

for i in imagelist:
    assert os.path.isfile(i)

    annotname = os.path.join(i.replace(f'.{ext}', '.txt'))
    if not os.path.isfile(annotname):
        sh.move(i, os.path.join(rootpath, "non_annotations", f"{os.path.basename(i)}"))

    else:
        with open(annotname, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                sh.move(i, os.path.join(rootpath, "non_annotations", f"{os.path.basename(i)}"))
                sh.move(annotname, os.path.join(rootpath, "non_annotations", f"{os.path.basename(annotname)}"))
