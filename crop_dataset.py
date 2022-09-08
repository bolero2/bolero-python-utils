import cv2
import os
from glob import glob


def crop_dataset(imglist, save_path):
    gap = 4
    rate = 1 / gap

    for imgname in imglist:
        img = cv2.imread(imgname)
        print(imgname)
        ih, iw, ic = img.shape
        size_h = int(ih / gap)
        size_w = int(iw / gap)

        txtname = os.path.basename(imgname).split('.')[0] + ".txt"
        dirs = os.path.dirname(imgname)
        f = open(os.path.join(dirs, txtname), 'r')
        lines = f.readlines()
        f.close()

        for r in range(gap):
            for c in range(gap):
                new_img = img[r * size_h:(r + 1) * size_h, c * size_w:(c + 1) * size_w,:]
                new_name = f"{os.path.basename(imgname).split('.')[0]}_{r}_{c}"
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(os.path.join(save_path, new_name + ".jpg"), new_img)

                print(f"\n\nc={c} [{c*rate}~{(c+1)*rate}] | r={r} [{r*rate}~{(r+1)*rate}]")
                print(f"Saved image name : {os.path.join(save_path, new_name + '.jpg')}")
                nih, niw, nic = new_img.shape
                new_annot = open(f"{os.path.join(save_path, new_name + '.txt')}", 'w')

                for line in lines:
                    line = line[:-1] if line[-1] == '\n' else line

                    class_id = line.split()[0]
                    bboxes = line.split()[1:]

                    center_x = float(bboxes[0])
                    center_y = float(bboxes[1])
                    width = float(bboxes[2])
                    height = float(bboxes[3])

                    if (r * rate < center_y < (r + 1) * rate) and (c * rate < center_x < (c + 1) * rate):

                        new_center_x = (center_x - (c * rate)) * gap
                        new_center_y = (center_y - (r * rate)) * gap
                        new_width = width * gap
                        new_height = height * gap

                        new_annot.write(f"{class_id} {new_center_x} {new_center_y} {new_width} {new_height}\n")

                new_annot.close()

if __name__ == "__main__":
    img_rootpath = "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/WIDER_total/total"
    imglist = glob(os.path.join(img_rootpath, "*.jpg"))

    crop_dataset(imglist, "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/Wider_total_cropped")
