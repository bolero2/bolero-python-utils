import cv2
import os
from glob import glob
from uuid import uuid1, uuid4


class convert_coordinate(object):
    def __init__(
        self, coordinate, img_shape, input_coord, input_type, output_coord, output_type
    ):
        # img_shape = (height, width) 이미지의 세로/가로
        """
        Arguments:
        1) coordinate : bbox coordinate
        2) img_shape = (height, width)
        3) input_coord = ['ccwh', 'xyrb', 'xywh']
        4) input_type = ['relat', 'abs']
        5) output_coord = ['ccwh', 'xyrb', 'xywh']
        6) output_type = ['relat', 'abs']
        return:
        converted coordinate, self.result
        example:
        answer = convert_coordinate(bbox, img_shape=(480, 640),
                                    input_coord='ccwh', input_type='relat',
                                    output_coord='xyrb', output_type='abs')
        """

        self.coord = [
            float(coordinate[0]),
            float(coordinate[1]),
            float(coordinate[2]),
            float(coordinate[3]),
        ]
        self.ih = int(img_shape[0])
        self.iw = int(img_shape[1])
        self.input_coord = input_coord
        self.input_type = input_type
        self.output_coord = output_coord
        self.output_type = output_type

        self.converted_type = self.convert_type(
            self.coord, self.input_type, self.output_type
        )
        self.converted_coord = self.convert_coord(
            self.converted_type, self.input_coord, self.output_coord
        )
        self.result = list(map(float, self.converted_coord)) if output_type == 'relat' else list(map(int, self.converted_coord))

    def __call__(self):
        return self.result

    def convert_type(self, coord, input_type, output_type):
        """
        coordinate type만 변경.
        1) relat(상대 좌표) to abs(절대 좌표)
        2) abs(절대 좌표) to relat(상대 좌표)
        """
        result = []

        if input_type != output_type:
            if input_type == "relat" and output_type == "abs":
                result = [
                    int(coord[0] * self.iw),
                    int(coord[1] * self.ih),
                    int(coord[2] * self.iw),
                    int(coord[3] * self.ih),
                ]
            elif input_type == "abs" and output_type == "relat":
                result = [
                    round(coord[0] / self.iw, 5),
                    round(coord[1] / self.ih, 5),
                    round(coord[2] / self.iw, 5),
                    round(coord[3] / self.ih, 5),
                ]

        elif input_type == output_type:
            result = coord

        return result

    def convert_coord(self, coord, input_coord, output_coord):
        """
        coordinate system을 변경.
        [대상 인자]
        1) ccwh : center_x, center_y, width, height
        2) xyrb : xmin, ymin, xmax, ymax
        3) xywh : xmin, ymin, width, height
        """
        result = []

        if input_coord != output_coord:
            if input_coord == "ccwh":
                center_x = coord[0]
                center_y = coord[1]
                width = coord[2]
                height = coord[3]

                if output_coord == "xywh":
                    result = [
                        center_x - (width / 2),
                        center_y - (height / 2),
                        width,
                        height,
                    ]
                elif output_coord == "xyrb":
                    result = [
                        center_x - (width / 2),
                        center_y - (height / 2),
                        center_x + (width / 2),
                        center_y + (height / 2),
                    ]
            elif input_coord == "xywh":
                xmin = coord[0]
                ymin = coord[1]
                width = coord[2]
                height = coord[3]

                if output_coord == "ccwh":
                    result = [xmin + (width / 2), ymin + (height / 2), width, height]
                elif output_coord == "xyrb":
                    result = [xmin, ymin, xmin + width, ymin + height]

            elif input_coord == "xyrb":
                xmin = coord[0]
                ymin = coord[1]
                xmax = coord[2]
                ymax = coord[3]

                width = xmax - xmin
                height = ymax - ymin

                if output_coord == "ccwh":
                    result = [xmin + (width / 2), ymin + (height / 2), width, height]
                elif output_coord == "xywh":
                    result = [xmin, ymin, width, height]

        elif input_coord == output_coord:
            result = coord

        return result


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


def get_hash():
    id1 = str(uuid1()).split('-')[0]
    id2 = str(uuid4()).split('-')[0]

    return id1 + id2


if __name__ == "__main__":
    img_rootpath = "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/WIDER_total/total"
    imglist = glob(os.path.join(img_rootpath, "*.jpg"))

    crop_dataset(imglist, "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/Wider_total_cropped")
