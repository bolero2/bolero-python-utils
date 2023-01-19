from datetime import datetime
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image
import torch
from time import time
import copy
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import getpass

MODEL_PATH = f"/home/{getpass.getuser()}/bolero/gopizza-ai-bolero2/yolov5"
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
PROJECT_HOME = os.getenv("PROJECT_HOME")

sys.path.append(MODEL_PATH)
# sys.path.append(os.path.join(MODEL_PATH, "network"))
sys.path.append(PYTHON_UTILS)
sys.path.append(PROJECT_HOME)

from utils.datasets import letterbox
from utils.general import box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from get_model import get_model
from bjob import Project
from metrics import get_metric_for_detection
from bdataset import DatasetParser
from bcommon import convert_coordinate as cc
# matplotlib.use('TkAgg')


def inference(imagelist=[]):
    agent = Project(config=f"/home/{getpass.getuser()}/bolero/projects/20221031-cbc770385205eb57/20221031-cbc770385205eb57.json")
    model = get_model(agent)

    weight_fpath = agent.weight_fpath(agent.hash)
    assert os.path.isfile(weight_fpath)

    is_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if is_cuda else "cpu")

    model = torch.load(weight_fpath)
    model = model.to(_device)
    model = model.eval()

    opt = model.yaml['test']
    opt['classes'] = None
    opt['iou_thres'] = 0.5
    opt['conv_thres'] = 0.5
    opt['image_size'] = (640, 640)

    """
    half = _device.type != 'cpu'  # half precision only supported on CUD
    if half:
        model.half()  # to FP16
    """

    pred_output = []

    for i, imgfile in tqdm(enumerate(imagelist), total=len(imagelist)):
        start_time = time()
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        ori_img = copy.deepcopy(img)
        ih, iw, ic = img.shape
        img = letterbox(img, opt['image_size'], stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(_device)

        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt['augment'])[0]
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
        pred_result = []

        for i, det in enumerate(pred):
            gn = torch.tensor((ih, iw, ic))[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det) != 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (ih, iw, ic)).round()

                # Print results
                # for c in set(det[:, -1].detach().cpu().numpy().tolist()):
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls.detach().cpu())
                    conf = float(conf.detach().cpu())
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    pred_result.append([cls, xywh[0], xywh[1], xywh[2], xywh[3], conf])

                    get_xyrb = cc(xywh, (ih, iw), 'ccwh', 'relat', 'xyrb', 'abs')
                    xyrb = get_xyrb()
                    xyrb = list(map(int, xyrb))

                    cv2.rectangle(ori_img, (xyrb[0], xyrb[1]), (xyrb[2], xyrb[3]), (0, 0, 255), 2)
                    # cv2.imshow("result", ori_img)
                    # cv2.waitKey(0)
                    # pred_result.append(xyrb)

        pred_output.append(pred_result)

        running_time = time() - start_time
        # print(f"Detection time : {np.round(running_time, 3)}")
        """
        cv2.imshow("aa", ori_img)
        if cv2.waitKey(0) == ord('q'):
            break
        """
    return pred_output


if __name__ == "__main__":
    # live Video version
    # inference_dough(camid=0)

    # Image version
    """
    root_path = f"/home/{getpass.getuser()}/bolero/dataset/sauce_segmentation_datasets/sauce_8class_train/only_sauce"
    test_images = sorted(glob(os.path.join(root_path, "images/*.jpg")))
    test_result = inference_dough(imagelist=test_images)

    savedir = os.path.join(root_path, "cropped")

    for ti, tr in zip(test_images, test_result):
        basename = os.path.splitext(os.path.basename(ti))[0]
        img = cv2.imread(ti, cv2.IMREAD_COLOR)
        annotfile = os.path.join(root_path, "annotations", os.path.splitext(os.path.basename(ti))[0] + ".png")
        assert os.path.isfile(annotfile)

        ih, iw, ic = img.shape

        margin_x = int(iw * 0.05)
        margin_y = int(ih * 0.05)
        # print("Margin :", [margin_x, margin_y])

        annot = Image.open(annotfile)

        for r in tr:
            xmin, ymin, xmax, ymax = r
            xmin = int(xmin - margin_x) if int(xmin - margin_x) > 0 else 0
            ymin = int(ymin - margin_y) if int(ymin - margin_y) > 0 else 0
            xmax = int(xmax + margin_x) if int(xmax + margin_x) < iw else iw
            ymax = int(ymax + margin_y) if int(ymax + margin_y) < ih else ih

            # cropped_annot = annot.crop((xmin, ymin, xmax - xmin, ymax - ymin))
            cropped_annot = annot.crop((xmin, ymin, xmax, ymax))
            cropped_image = img[ymin:ymax, xmin:xmax, :]
            sum_uniq = np.sum(np.unique(cropped_annot))

            if sum_uniq != 0:
                already_saved = 0
                saved_imagepath = glob(os.path.join(savedir, "images"))
                for si in saved_imagepath:
                    if basename in os.path.basename(si):
                        already_saved += 1

                basename = basename + "_" + str(already_saved)
                cv2.imwrite(os.path.join(savedir, "images", f"{basename}.jpg"), cropped_image)
                cropped_annot.save(os.path.join(savedir, "annotations", f"{basename}.png"))

            # cv2.write(os.path.join(savedir, "images", f"{basename}.jpg", cropped_image))
            # cropped_annot.save(os.path.join(savedir, "annotations", f"{basename}.png"))

            # plt.subplot(1, 2, 1)
            # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # plt.subplot(1, 2, 2)
            # plt.imshow(cropped_annot)
            # plt.show()
    """

if __name__ == "__main__":
    category_list = ['dough', 'pepperoni']
    agent = Project(config=f"/home/{getpass.getuser()}/bolero/projects/20221031-cbc770385205eb57/20221031-cbc770385205eb57.json")
    pkl_file = os.path.join(agent.save_dir, f"train_{agent.hash}.pkl")
    test_dataset = DatasetParser.load_dataset(pkl_path=pkl_file)

    test_images, test_gts, _, _ = test_dataset
    test_images = test_images
    test_gts = test_gts

    pred_output = inference(test_images)
    df_list = []

    for ti, tg, td in zip(test_images, test_gts, pred_output):
        imagename = os.path.abspath(ti)
        gt_class = [category_list[int(x[0])] for x in tg]
        gt_bbox = [x[1:] for x in tg]
        dt_class = [category_list[int(x[0])] for x in td]
        dt_bbox = [x[1:5] for x in td]
        dt_conf = [x[5] for x in td]

        df_list.append([imagename, imagename, gt_class, gt_bbox, dt_class, dt_bbox, dt_conf])

    result_df = pd.DataFrame(df_list, columns=["input_data", "results_path", "true_y_classname", "true_y_bndboxes", "predicted_y_classname", "predicted_y_bndboxes", "confidence"])
    get_metric_for_detection(result_df, csv_path=os.path.join(agent.save_dir, "test-result.csv"), category_list=category_list)
