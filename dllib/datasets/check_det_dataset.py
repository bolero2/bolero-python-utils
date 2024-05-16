import os
from glob import glob
from random import shuffle
import shutil as sh
from tqdm import tqdm


rootpath = '/home/bulgogi/bolero/dataset/dsc_dataset/original/instance_segmentation/total'
ext = 'jpg'

imagelist = glob(os.path.join(rootpath, "images", f"*.{ext}"))

shuffle(imagelist)
os.makedirs(os.path.join(rootpath, "non_annotations"), exist_ok=True)

for idx, i in enumerate(tqdm(imagelist, total=len(imagelist))):
    assert os.path.isfile(i)

    annotname = os.path.join(i.replace(f'.{ext}', '.txt').replace("/images/", "/annotations/"))
    if not os.path.isfile(annotname):
        sh.move(i, os.path.join(rootpath, "non_annotations", f"{os.path.basename(i)}"))

    else:
        with open(annotname, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                sh.move(i, os.path.join(rootpath, "non_annotations", f"{os.path.basename(i)}"))
                sh.move(annotname, os.path.join(rootpath, "non_annotations", f"{os.path.basename(annotname)}"))
