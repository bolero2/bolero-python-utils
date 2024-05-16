from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil as sh
from PIL import Image
import random
import os

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

# 기존 데이터셋 개수의 {IMAGE_COUNT} 배수만큼 만들 것.
IMAGE_COUNT = 2

rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped'
savepath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped/augmented/mosaic'
os.makedirs(savepath, exist_ok=True)
os.makedirs(os.path.join(savepath, "images"), exist_ok=True)
os.makedirs(os.path.join(savepath, "annotations"), exist_ok=True)

imglist = glob(os.path.join(rootpath, 'images', '*.jpg'))
sample_image = cv2.imread(imglist[0], cv2.IMREAD_COLOR)
ih, iw, ic = sample_image.shape
frag_size = (int(ih / 2), int(iw / 2), ic)
fh, fw, fc = frag_size
aug_count = 0
print(ih, iw, ic)
print(frag_size)

cmap = get_colormap()
palette = [value for color in cmap for value in color]

for i in range(len(imglist) * IMAGE_COUNT):    # 기존 데이터셋의 2배수만큼 만들 것
    image_canvas = np.zeros_like(sample_image)
    is_filled = [False, False, False, False]
    sectors = [[0, 0], [0 + fh, 0], [0, 0 + fw], [0 + fh, 0 + fw]]
    before_index = -1
    
    colmap_canvas = Image.new('P', (iw, ih))
    while not all(is_filled):
        target_index = random.randint(0, len(imglist) - 1)
        
        if before_index == target_index:
            continue

        target_file = imglist[target_index]
#         print(target_file)
        img = cv2.imread(target_file, cv2.IMREAD_COLOR)
        target_colmapname = target_file.replace("images", "annotations").replace(".jpg", ".png")
        colmap = Image.open(target_colmapname)
        
        start_ih = random.randint(0, int(ih - fh - 1))
        start_iw = random.randint(0, int(iw - fw - 1))
        
        frag_image = img[start_ih:start_ih + fh, start_iw:start_iw + fw, :]
        frag_colmap = colmap.crop((start_iw, start_ih, start_iw + fw, start_ih + fh))
        
        unique, counts = np.unique(np.array(frag_colmap), return_counts=True)
        uniq_counts_dict = dict(zip(unique, counts))
        total_count = sum(counts.tolist())
        nonlabel_percentage = float(uniq_counts_dict[0] / total_count)
        if nonlabel_percentage >= 0.93:
            continue
        
        for idx, boolval in enumerate(is_filled):
            if boolval is False:
                start_point = sectors[idx]
                image_canvas[start_point[0]:start_point[0] + fh, start_point[1]:start_point[1] + fw, :] = frag_image
                colmap_canvas.paste(frag_colmap, (start_point[1], start_point[0]))
                is_filled[idx] = True
                break
        before_index = target_index
    colmap_canvas.putpalette(palette)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_canvas)
#     plt.subplot(1, 2, 2)
#     plt.imshow(colmap_canvas)
#     plt.show()
    
    cv2.imwrite(os.path.join(savepath, "images", f"augmented_mosaic_{i}.jpg"), image_canvas)
    colmap_canvas.save(os.path.join(savepath, "annotations", f"augmented_mosaic_{i}.png"))
