from glob import glob
from random import shuffle
import os
import shutil as sh


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('-p', '--path', help='dataset path', type=str)

    args = parser.parse_args()

    args.path= os.path.abspath(args.path)

    print("Args :", args, "\n\n\n")

    return args


if __name__ == "__main__":
    args = parse_args()

    # rootpath = '/home/bulgogi/bolero/dataset/aistt_ingredients/v1/tomato_sauce/train_dataset'
    # rootpath = '/home/bulgogi/bolero/dataset/aistt_ins_seg_dataset/total/train_dataset_dscp+basil+marinated'

    rootpath = args.path
    print("Root path :", rootpath)

    imglist = glob(os.path.join(rootpath, 'images', '*.jpg'))
    shuffle(imglist)

    print("Image Count :", len(imglist))
    count = len(imglist)
    save_target = ['train', 'valid']
    save_rate = [0.95, 0.05]

    for i in range(count):
        t_image = imglist[i]
        t_annot = os.path.join(rootpath, "labels", os.path.splitext(os.path.basename(t_image))[0] + ".txt")

        assert os.path.isfile(t_image) and os.path.isfile(t_annot)

        if i < int(count * save_rate[0]):
            savepath = 'train'

        elif int(count * save_rate[0]) <= i < int(count * sum(save_rate[0:2])):
            savepath = 'valid'

        if 'test' in save_target and len(save_rate) > 2:
            if int(count * sum(save_rate[0:3])) <= i:
                savepath = 'test'

        if not os.path.isdir(os.path.join(rootpath, savepath)):
            os.makedirs(os.path.join(rootpath, savepath), exist_ok=True)

        if not os.path.isdir(os.path.join(rootpath, savepath, "images")):
            os.makedirs(os.path.join(rootpath, savepath, "images"), exist_ok=True)
        if not os.path.isdir(os.path.join(rootpath, savepath, "labels")):
            os.makedirs(os.path.join(rootpath, savepath, "labels"), exist_ok=True)

        sh.copy(t_image, os.path.join(rootpath, savepath, "images"))
        sh.copy(t_annot, os.path.join(rootpath, savepath, "labels"))
