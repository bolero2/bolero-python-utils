import os
from glob import glob
import cv2
import shutil as sh


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/bolero/dataset/aistt_ins_seg_dataset/v2/total/images'
    savedir = os.path.join(dataset_rootpath, 'selected')
    trashbox = os.path.join(dataset_rootpath, '_trashbox')

    if not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
    if not os.path.isdir(trashbox):
        os.makedirs(trashbox, exist_ok=True)

    imglist = sorted(glob(os.path.join(dataset_rootpath, '*.jpg')))
    imgcount = len(imglist)
    index = 0

    while True:
        filename = imglist[index]
        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        cv2.imshow("image", img)
        key = cv2.waitKey(0)

        if key == ord('a'):
            index -= 1
        elif key == ord('d'):
            index += 1
        elif key == ord('s'):
            sh.copy(filename, savedir)
            index += 1
            # os.remove(filename)
            # del imglist[imglist.index(filename)]
            # imgcount = len(imglist)
        elif key == ord('r'):
            sh.copy(filename, savedir)
            del imglist[imglist.index(filename)]
            index += 1
            imgcount = len(imglist)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

        if index < 0:
            print("First Image! ====================")
            index = 0
        if index > imgcount - 1:
            print("==================== Last Image!")
            index = imgcount - 1


        if index % 100 == 0:
            print("Now Index :", index)
