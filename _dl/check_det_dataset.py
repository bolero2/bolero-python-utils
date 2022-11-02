import os
from glob import glob
from random import shuffle
import shutil as sh


imagelist = glob("images/*.jpg")
shuffle(imagelist)

for i in imagelist:
    annotname = os.path.join("annotations", os.path.basename(i).replace('.jpg', '.txt'))

    assert os.path.isfile(annotname), f"{annotname} is not file."
    assert os.path.isfile(i)

    with open(annotname, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            sh.move(i, f"non_annotations/{os.path.basename(i)}")
            sh.move(annotname, f"non_annotations/{os.path.basename(annotname)}")

