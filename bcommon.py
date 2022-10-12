import cv2
import os
import numpy as np
from glob import glob
from uuid import uuid1, uuid4
from tqdm import tqdm


def image_compositing(src1, src2, resize=None):
    if isinstance(src1, str) and os.path.isfile(src1):
        src1 = cv2.imread(src1, cv2.IMREAD_COLOR)# / 255.0
        if resize is not None:
            src1 = cv2.resize(src1, tuple(resize))
    if isinstance(src2, str) and os.path.isfile(src2):
        src2 = cv2.imread(src2, cv2.IMREAD_COLOR)# / 255.0
        if resize is not None:
            src2 = cv2.resize(src2, tuple(resize))

    alpha = 0.5

    image = cv2.addWeighted(src1, alpha, src2, (1 - alpha), 0)
    #image = image * 255.0

    return image


def compositing(colormap_path, original_path, cmap_format='png', original_format='jpg', resize=(960, 960)):
    cmap_list = glob(os.path.join(colormap_path, f'*.{cmap_format}'))
    savepath = os.path.join(colormap_path, 'composited')
    os.makedirs(savepath, exist_ok=True)
    print("\n\n -> save path :", savepath, "\n\n")

    for i, c in tqdm(enumerate(cmap_list), total=len(cmap_list)):
        ori_image = os.path.join(original_path, os.path.basename(c).replace(cmap_format, original_format))
        assert os.path.isfile(ori_image)

        output = image_compositing(c, ori_image, resize=resize)

        cv2.imwrite(os.path.join(savepath, f"composited_{os.path.basename(c)}"), output)


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

def get_colormap(count=256, cmap_type="pascal"):
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """

    def bit_get(val, idx):
        """Gets the bit value.
        Args:
            val: Input value, int or numpy int array.
            idx: Which bit of the input val.
        Returns:
            The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    cmap_type = cmap_type.lower()
    if cmap_type == "voc" or cmap_type == "coco":
        cmap_type = "pascal"

    if cmap_type == "pascal":
        colormap = np.zeros((count, 3), dtype=int)
        ind = np.arange(count, dtype=int)

        for shift in reversed(list(range(8))):
            for channel in range(3):
                colormap[:, channel] |= bit_get(ind, channel) << shift
            ind >>= 3

        return colormap

    elif cmap_type == "cityscape":
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]

        return colormap

    elif cmap_type == "ade20k":
        return np.asarray(
            [
                [0, 0, 0],
                [120, 120, 120],
                [180, 120, 120],
                [6, 230, 230],
                [80, 50, 50],
                [4, 200, 3],
                [120, 120, 80],
                [140, 140, 140],
                [204, 5, 255],
                [230, 230, 230],
                [4, 250, 7],
                [224, 5, 255],
                [235, 255, 7],
                [150, 5, 61],
                [120, 120, 70],
                [8, 255, 51],
                [255, 6, 82],
                [143, 255, 140],
                [204, 255, 4],
                [255, 51, 7],
                [204, 70, 3],
                [0, 102, 200],
                [61, 230, 250],
                [255, 6, 51],
                [11, 102, 255],
                [255, 7, 71],
                [255, 9, 224],
                [9, 7, 230],
                [220, 220, 220],
                [255, 9, 92],
                [112, 9, 255],
                [8, 255, 214],
                [7, 255, 224],
                [255, 184, 6],
                [10, 255, 71],
                [255, 41, 10],
                [7, 255, 255],
                [224, 255, 8],
                [102, 8, 255],
                [255, 61, 6],
                [255, 194, 7],
                [255, 122, 8],
                [0, 255, 20],
                [255, 8, 41],
                [255, 5, 153],
                [6, 51, 255],
                [235, 12, 255],
                [160, 150, 20],
                [0, 163, 255],
                [140, 140, 140],
                [250, 10, 15],
                [20, 255, 0],
                [31, 255, 0],
                [255, 31, 0],
                [255, 224, 0],
                [153, 255, 0],
                [0, 0, 255],
                [255, 71, 0],
                [0, 235, 255],
                [0, 173, 255],
                [31, 0, 255],
                [11, 200, 200],
                [255, 82, 0],
                [0, 255, 245],
                [0, 61, 255],
                [0, 255, 112],
                [0, 255, 133],
                [255, 0, 0],
                [255, 163, 0],
                [255, 102, 0],
                [194, 255, 0],
                [0, 143, 255],
                [51, 255, 0],
                [0, 82, 255],
                [0, 255, 41],
                [0, 255, 173],
                [10, 0, 255],
                [173, 255, 0],
                [0, 255, 153],
                [255, 92, 0],
                [255, 0, 255],
                [255, 0, 245],
                [255, 0, 102],
                [255, 173, 0],
                [255, 0, 20],
                [255, 184, 184],
                [0, 31, 255],
                [0, 255, 61],
                [0, 71, 255],
                [255, 0, 204],
                [0, 255, 194],
                [0, 255, 82],
                [0, 10, 255],
                [0, 112, 255],
                [51, 0, 255],
                [0, 194, 255],
                [0, 122, 255],
                [0, 255, 163],
                [255, 153, 0],
                [0, 255, 10],
                [255, 112, 0],
                [143, 255, 0],
                [82, 0, 255],
                [163, 255, 0],
                [255, 235, 0],
                [8, 184, 170],
                [133, 0, 255],
                [0, 255, 92],
                [184, 0, 255],
                [255, 0, 31],
                [0, 184, 255],
                [0, 214, 255],
                [255, 0, 112],
                [92, 255, 0],
                [0, 224, 255],
                [112, 224, 255],
                [70, 184, 160],
                [163, 0, 255],
                [153, 0, 255],
                [71, 255, 0],
                [255, 0, 163],
                [255, 204, 0],
                [255, 0, 143],
                [0, 255, 235],
                [133, 255, 0],
                [255, 0, 235],
                [245, 0, 255],
                [255, 0, 122],
                [255, 245, 0],
                [10, 190, 212],
                [214, 255, 0],
                [0, 204, 255],
                [20, 0, 255],
                [255, 255, 0],
                [0, 153, 255],
                [0, 41, 255],
                [0, 255, 204],
                [41, 0, 255],
                [41, 255, 0],
                [173, 0, 255],
                [0, 245, 255],
                [71, 0, 255],
                [122, 0, 255],
                [0, 255, 184],
                [0, 92, 255],
                [184, 255, 0],
                [0, 133, 255],
                [255, 214, 0],
                [25, 194, 194],
                [102, 255, 0],
                [92, 0, 255],
            ]
        )
    elif cmap_type == "kvasir":
        return np.asarray(
            [
                [0, 0, 0],
                [255, 255, 255]
            ]
        )

    else:
        raise ValueError("Unsupported dataset.")


if __name__ == "__main__":
    """
    img_rootpath = "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/WIDER_total/total"
    imglist = glob(os.path.join(img_rootpath, "*.jpg"))

    crop_dataset(imglist, "/Users/bolero/dc/dataset/WiderFaceDetectionDataset/Wider_total_cropped")
    """
    root_path = '/home/bulgogi/bolero/dataset/sauce_segmentation_datasets/sauce_8class_train/original/no_dough/images'
    colormap_path = os.path.join(root_path, 'outputs')
    original_path = root_path

    compositing(colormap_path, original_path, resize=(2048, 1024))
    # """
