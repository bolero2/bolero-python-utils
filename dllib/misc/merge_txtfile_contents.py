import os
from glob import glob


if __name__ == "__main__":
    targetpath = "/home/bulgogi/bolero/dataset/dsc_dataset/original/instance_segmentation/total/pepperoni_det_result/final/labels/*.txt"
    savepath = "/home/bulgogi/bolero/dataset/dsc_dataset/original/instance_segmentation/total/annotations"
    targets = glob(targetpath)

    for t in targets:
        basename = os.path.basename(t)
        save_target = os.path.join(savepath, basename)
        if not os.path.isfile(save_target):
            print(save_target, "is not txtfile!")

        target_f = open(t, 'r')
        lines = target_f.readlines()

        f = open(save_target, "a+")
        f.writelines(lines)

        target_f.close()
        f.close()
