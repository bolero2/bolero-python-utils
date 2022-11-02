import os
from glob import glob
import shutil as sh


path = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped/from_only_tomato_sauce/images'
annotpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped/from_only_tomato_sauce/annotations'
# annotpath = path
imglist = glob(os.path.join(path, "*.jpg"))

for index, i in enumerate(imglist):
    basename = os.path.basename(i)
    dirname = os.path.dirname(i)

    annotfile = os.path.join(annotpath, basename.replace('.jpg', '.png'))

    os.rename(i, os.path.join(dirname, f"tomato_sauce_{index}.jpg"))
    os.rename(annotfile, os.path.join(dirname, f"tomato_sauce_{index}.png"))
