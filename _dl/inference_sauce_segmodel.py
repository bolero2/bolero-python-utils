from datetime import datetime
import os
import sys 
import cv2
import numpy as np
from PIL import Image
import onnx
import onnxruntime
import time
import copy
from matplotlib import pyplot as plt 
from glob import glob
import matplotlib
import math
import getpass
import yaml
from tabulate import tabulate as tb

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
from commons import get_colormap

COLORMAP = get_colormap(256)
CMAP_LIST = COLORMAP.tolist()
PALETTE = [value for color in CMAP_LIST for value in color]


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)

    return model

def tensor_transform(tensor, image_size=(192, 336)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tensor = tensor.type(torch.float32)
    tensor = tensor.permute(2, 0, 1)

    _transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean, std),
    ])

    tensor = _transform(tensor).unsqueeze(0)
    return tensor

def numpy_transform(image: np.ndarray):
    image = image[:, :, ::-1]
    image = image.astype(np.float32)
    image = image / 255.0
    return image

def inference_segmentation(target, image_size=[336, 192], include_labels=[0, 1]):
    global COLORMAP
    global PALETTE
    global segmentation_model

    segmodel = segmentation_model

    assert segmodel is not None, "Segmentation model is NoneType."

    ori_target = copy.deepcopy(target)

    target = numpy_transform(target)
    canvas = np.zeros(tuple([*(image_size[::-1]), 3])).astype(np.uint8)

    pred = None

    try:
        with torch.no_grad():
            target = torch.from_numpy(target).to(_device)
            target = tensor_transform(target, [*(image_size[::-1])])
            pred = segmodel(target)
            pred = F.interpolate(pred, size=target.size()[-2:],
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            pred = pred.astype(np.uint8)

        pred = np.where(np.isin(pred, include_labels), pred, 0)

        for i, color in enumerate(COLORMAP):
            for j in range(3):
                canvas[:, :, j][pred == i] = COLORMAP[i][j]

    except Exception as e:
        print("Exception is occurred :", e)

    return pred, canvas

if __name__ == "__main__":
    SEGMENTATION_MODEL_PATH = f"/home/{getpass.getuser()}/bolero/gopizza-ai-bolero2/sandbox/PIDNet"
    assert os.path.isdir(SEGMENTATION_MODEL_PATH)
    print("Segmentation Model fpath :", SEGMENTATION_MODEL_PATH)
    sys.path.append(SEGMENTATION_MODEL_PATH)

    import models as segmodel

    segmentation_model = segmodel.pidnet.get_pred_model('pidnet-s', 7)
    del sys.modules['models']
    del sys.path[sys.path.index(SEGMENTATION_MODEL_PATH)]
    del segmodel
    if segmentation_model is not None:
        print("Segmentation model is loaded successfully.")

    PYTHON_UTILS = os.getenv("PYTHON_UTILS")
    PROJECT_HOME = os.getenv("PROJECT_HOME")

    sys.path.append(PYTHON_UTILS)
    sys.path.append(PROJECT_HOME)

    from bjob import Project
    from commons import convert_coordinate as cc
    from commons import get_colormap, decode_seg_map_sequence

    COLORMAP = get_colormap(255)
    CMAP_LIST = COLORMAP.tolist()
    PALETTE = [value for color in CMAP_LIST for value in color]

    segweight_fpath = f"/home/{getpass.getuser()}/bolero/gopizza-ai-bolero2/AISTT/seg_336192.pt"
    segmentation_model = load_pretrained(segmentation_model, segweight_fpath)

    is_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if is_cuda else "cpu")

    segmentation_model = segmentation_model.to(_device)
    segmentation_model = segmentation_model.eval()

    rootpath = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped/images'
    imglist = glob(os.path.join(rootpath, "*.jpg"))

    for i in imglist:
        img = cv2.imread(i,cv2.IMREAD_COLOR)
        ih, iw, ic = img.shape
        basename = os.path.basename(i)
        savedir = '/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/seg_data/cropped/sauce_inference'

        seg_pred, seg_colormap = inference_segmentation(img, include_labels=[0, 5])
        if np.unique(seg_pred).tolist() == [0]:
            continue

        cv2.imshow("original", img)
        cv2.imshow("seg colormap", seg_colormap)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('s'):
            img_png = Image.fromarray(seg_pred).convert('P')
            img_png = img_png.resize((iw, ih))
            img_png.putpalette(PALETTE)
            img_png.save(os.path.join(savedir, basename.replace(".jpg", ".png")))
