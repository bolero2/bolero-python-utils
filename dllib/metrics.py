import os
import pandas as pd
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import math
import cv2
import datetime
from PIL import Image
import time
from decimal import Decimal as dec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score # Clustering 용
from tabulate import tabulate
from tqdm import tqdm
from commons import get_colormap, convert_coordinate
import logging
logger = logging.getLogger(__name__)

np.seterr(all="ignore")\

def iou(box1, box2):
    def _is_box_intersect(box1, box2):
        if (
            abs(box1[0] - box2[0]) < box1[2] + box2[2]
            and abs(box1[1] - box2[1]) < box1[3] + box2[3]
        ):
            return True
        else:
            return False

    def _get_area(box):
        return box[2] * box[3]

    def _get_intersection_area(box1, box2):
    # intersection area
        return abs(max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])) * abs(
            max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3])
        )
    def _get_union_area(box1, box2, inter_area=None):
        area_a = _get_area(box1)
        area_b = _get_area(box2)
        if inter_area is None:
            inter_area = _get_intersection_area(box1, box2)

        return float(area_a + area_b - inter_area)

    # if boxes dont intersect
    if _is_box_intersect(box1, box2) is False:
        # print("zero")
        return 0
    inter_area = _get_intersection_area(box1, box2)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    # intersection over union
    iou = inter_area / union
    if iou < 0:
        iou = 0
    # print(f"iou: {iou}")
    # assert iou >= 0
    return iou

def AP(precision_list, recall_list):
    start_index = 0
    total_area = 0
    # print(precision_list, recall_list)
    for recall_index in range(len(recall_list) - 1):
        if recall_list[recall_index] == recall_list[recall_index + 1]:
            if start_index == 0:
                width = recall_list[recall_index]
            else:
                width = dec(str(recall_list[recall_index])) - dec(
                    str(recall_list[start_index])
                )
            start_index = recall_index
            height = precision_list[recall_index]
            total_area = total_area + dec(str(width)) * dec(str(height))

    return total_area


##############


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        print(count.shape)

        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        print(gt_image.shape)
        print(pre_image.shape)
        if gt_image.shape != pre_image.shape:
            print("GT_Image's shape is different with PRE_IMAGE's shape!")
            exit(0)
        try:
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        except Exception as e:
            pass

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def get_average(data: list):
    if len(data) > 0:
        return sum(data) / len(data)
    else:
        return 0

def iou(box1, box2):
    def _is_box_intersect(box1, box2):
        if (
            abs(box1[0] - box2[0]) < box1[2] + box2[2]
            and abs(box1[1] - box2[1]) < box1[3] + box2[3]
        ):
            return True
        else:
            return False

    def _get_area(box):
        return box[2] * box[3]

    def _get_intersection_area(box1, box2):
    # intersection area
        return abs(max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])) * abs(
            max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3])
        )
    def _get_union_area(box1, box2, inter_area=None):
        area_a = _get_area(box1)
        area_b = _get_area(box2)
        if inter_area is None:
            inter_area = _get_intersection_area(box1, box2)

        return float(area_a + area_b - inter_area)

    # if boxes dont intersect
    if _is_box_intersect(box1, box2) is False:
        # print("zero")
        return 0
    inter_area = _get_intersection_area(box1, box2)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    # intersection over union
    iou = inter_area / union
    if iou < 0:
        iou = 0
    # print(f"iou: {iou}")
    # assert iou >= 0
    return iou


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)

        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # printShape(gt_image)
        # printShape(pre_image)
        if gt_image.shape != pre_image.shape:
            print("GT_Image's shape is different with PRE_IMAGE's shape!")
            exit(0)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def get_metric_for_detection(
    dataframe,
    csv_path,
    category_list,
    iou_threshold=0.4,
    csv_save_path=None,
    sorting_index=3,
    rootpath=""):

    # csv save -> image_path, Classname, Yt(label, cx, cy, w, h), Yp(label, cx, cy, w, h), Yp(conf)
    # rootpath = path_join(rootpath[0], rootpath[1])

    # 변수 선언하기
    sum_TP, sum_FP, sum_FN, iou_val = 0, 0, 0, 0

    # Dataframe 가져오기
    columns = dataframe.columns.values.tolist()
    columns = columns + ["avg_iou", "tp", "fp", "fn", "sum_tp", "sum_fp", "sum_fn", "precision", "recall"]
    df = dataframe.to_numpy()

    # 전체 항목 설정하기
    _test_image_list, result_image_list, gt_class_list, gt_bboxes_list, dt_class_list, dt_bboxes_list, dt_conf_list \
        = df[:, 0].tolist(), df[:, 1].tolist(), df[:, 2].tolist(), df[:, 3].tolist(), df[:, 4].tolist(), df[:, 5].tolist(), df[:, 6].tolist()

    assert ((len(_test_image_list) == len(gt_class_list)) and (len(gt_bboxes_list) == len(dt_class_list))) and (len(dt_bboxes_list) == len(dt_conf_list)), \
        "Result has difference count!"

    # Saving image paths : [absolute path (test_image_path)] is made from [rootpath + relative path (_test_image_list)]
    test_image_list = [os.path.abspath(os.path.join(rootpath, x)) for x in _test_image_list]
    assert len(test_image_list) == len(_test_image_list), \
        f"_test_image_list count {len(_test_image_list)} and test_image_list count {len(test_image_list)} is different."

    # _test_image_list = relative path (for csv saving)
    # test_image_list = absolute path (for read image, cv2.imread -> ih, iw, ic = image.shape)
    class_gt_count = {key: 0 for key in category_list}

    for class_list in gt_class_list:
        for class_name in class_list:
            class_gt_count[class_name] += 1

    total_gt_count = sum([val for _, val in class_gt_count.items()])
    total_dt_count = 0

    for dt in dt_class_list:
        if len(dt) == 1 and dt[0] == '':
            continue
        else:
            total_dt_count += len(dt)

    logger.info("\n\n ============== Metric START ==============\n\n")
    logger.info("-> Image, GT Bndboxes, DT Bndboxes Counts")
    table = [
        ["Getting Metric Started -> now Datetime", datetime.datetime.now()],
        ["Total Test-Images count", len(test_image_list)],
        ["Total Ground-Truth Bndboxes count", total_gt_count],
        ["Total Predicted Bndboxes count", total_dt_count],
        ["IoU Threshold", iou_threshold]
    ]
    print()
    print(tabulate(table, tablefmt='grid'), end='\n\n')

    ###########################################
    # get Total Object count
    ###########################################
    new_df = list()
    print()

    # image 갯수만큼 반복
    # Dataframe 재구성 : image1개, DT(label, bbox, conf) 1개, GT 전부다(Image1개에 해당되는 모든 것) + DT Conf로 정렬
    for _, (imgpath, gt_classes, gt_bboxes, dt_classes, dt_bboxes, dt_confs) \
        in tqdm(enumerate(zip(test_image_list, gt_class_list, gt_bboxes_list, dt_class_list, dt_bboxes_list, dt_conf_list)),
                               desc=' -> [Detection Metric - 1] Reconstruct Dataframe ',
                               total=len(test_image_list)):
        assert os.path.isfile(
            imgpath), f"Image path: {imgpath} is invalid path. Please check path."
        
        for _, (dt_label, dt_bbox, dt_conf) in enumerate(zip(dt_classes, dt_bboxes, dt_confs)):
            new_df.append([imgpath, dt_label, dt_bbox, dt_conf, gt_classes, gt_bboxes])

    if sorting_index != 0:
        new_df = sorted(new_df, key=lambda x: x[sorting_index], reverse=True)

    new_df = pd.DataFrame(new_df, columns=["input_data", "predicted_y_classname", "predicted_y_bndboxes", "confidence", "true_y_classname", "true_y_bndboxes"]).to_numpy()
    new_test_image_list, new_dt_class_list, new_dt_bboxes_list, new_dt_conf_list, new_gt_class_list, new_gt_bboxes_list \
        = new_df[:, 0].tolist(), new_df[:, 1].tolist(), new_df[:, 2].tolist(), new_df[:, 3].tolist(), new_df[:, 4].tolist(), new_df[:, 5].tolist()

    assert len(new_test_image_list) == len(new_dt_bboxes_list) == len(new_dt_class_list) == len(new_dt_conf_list) == len(new_gt_class_list) == len(new_gt_bboxes_list), \
        "dataframe row count is different!"

    total_TPFP_list = list()
    class_TPFP_list = {class_name: list() for class_name in category_list}
    already_founded_gt = list()

    """
    SUM TP : 3754
    SUM FP : 844
    Precision : 0.8164419312744672
    Recall : 0.8294299602297834
    AP : 0.7794775388083534606065724952
    F1 Score : 0.8228846996931171
    """
    print()
    for _, (imgpath, dt_label, dt_bbox, dt_conf, gt_classes, gt_bboxes) \
        in tqdm(enumerate(zip(new_test_image_list, new_dt_class_list, new_dt_bboxes_list, new_dt_conf_list, new_gt_class_list, new_gt_bboxes_list)),
                               desc=' -> [Detection Metric - 2] Calculate TP and FP ',
                               total=len(new_test_image_list)):
        assert os.path.isfile(imgpath), \
            f"Image path: {imgpath} is invalid path. Please check path."

        assert os.path.isfile(imgpath), "Invalid Image file path!"
        np_image = cv2.imread(imgpath)
        ih, iw, _ = np_image.shape
        del np_image

        TP, FP = 0, 0
        dt_iou_list = list()

        if dt_label == '' and dt_bbox == [] and dt_conf == -1:
            total_TPFP_list.append([imgpath, dt_label, dt_bbox, dt_conf, 0, 0, 0])

        elif dt_label != '' and dt_bbox != [] and dt_conf != -1:
            cvt_dt_coord = convert_coordinate(
                    dt_bbox, (ih, iw), "ccwh", "relat", "xywh", "abs")
            dt_coord = cvt_dt_coord()
            
            for gt_label, gt_bbox in zip(gt_classes, gt_bboxes):
                cvt_gt_coord = convert_coordinate(
                        gt_bbox, (ih, iw), "ccwh", "relat", "xywh", "abs")
                gt_coord = cvt_gt_coord()

                iou_val = float(iou(dt_coord, gt_coord))

                if (iou_val >= iou_threshold) and (dt_label == gt_label) and ([imgpath, gt_bbox, gt_label] not in already_founded_gt):
                    TP = 1
                    dt_iou_list.append(iou_val)
                    already_founded_gt.append([imgpath, gt_bbox, gt_label])

            if TP == 0:
                FP = 1
            
            dt_avg_iou = get_average(dt_iou_list)

            total_TPFP_list.append([imgpath, dt_label, dt_bbox, dt_conf, dt_avg_iou, TP, FP])
            class_TPFP_list[dt_label].append([imgpath, dt_label, dt_bbox, dt_conf, TP, FP])

    # print("LEN already_founded_gt :", len(already_founded_gt))
    TPFP_columns = ['input_data', 'predicted_y_classname', 'predicted_y_bndboxes', 'confidence', 'avg_iou', 'tp', 'fp']
    total_TPFP_df = pd.DataFrame(total_TPFP_list, columns=TPFP_columns)
    class_TPFP_df = [pd.DataFrame(class_TPFP_list[x], 
                                  columns=TPFP_columns[:TPFP_columns.index('avg_iou')] + TPFP_columns[TPFP_columns.index('avg_iou') + 1:]) \
                                      for x in category_list]

    np_TPFP_df = total_TPFP_df.to_numpy()
    TPFP_test_image_list, TPFP_dt_class_list, TPFP_dt_bboxes_list, TPFP_dt_conf_list, TPFP_dt_iou_list, TP_list, FP_list \
        = np_TPFP_df[:, 0].tolist(), np_TPFP_df[:, 1].tolist(), np_TPFP_df[:, 2].tolist(), np_TPFP_df[:, 3].tolist(), \
            np_TPFP_df[:, 4].tolist(), np_TPFP_df[:, 5].tolist(), np_TPFP_df[:, 6].tolist()

    sum_TP, sum_FP = 0, 0
    precision, recall = 0.0, 0.0
    total_precision_list, total_recall_list = list(), list()
    PR_list = list()
    print()

    # Get Precision and Recall
    for index, (TP_val, FP_val) in tqdm(enumerate(zip(TP_list, FP_list)),
                               desc=' -> [Detection Metric - 3] Calculate Precision/Recall ',
                               total=len(TPFP_test_image_list)):

        sum_TP = sum_TP + TP_val
        sum_FP = sum_FP + FP_val

        precision = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) != 0 else 0
        recall = sum_TP / total_gt_count if total_gt_count != 0 else 0

        total_precision_list.append(precision)
        total_recall_list.append(recall)

        PR_list.append([sum_TP, sum_FP, precision, recall])

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    total_PR_df = pd.DataFrame(PR_list, columns=['sum_tp', 'sum_fp', 'precision', 'recall'])
    total_TPFP_df = pd.concat([total_TPFP_df, total_PR_df], axis=1)

    # Get Precision and Recall, mAP usin class_TPFP_dataframe
    target_ap = {key: 0 for key in category_list}
    for cnum in range(len(category_list)):
        target_sum_TP, target_sum_FP = 0, 0

        target_PR_list = list()
        target_df = class_TPFP_df[cnum].to_numpy()

        target_test_image_list, target_dt_class_list, target_dt_bboxes_list, target_dt_conf_list, target_TP_list, target_FP_list \
            = target_df[:, 0].tolist(), target_df[:, 1].tolist(), target_df[:, 2].tolist(), target_df[:, 3].tolist(), target_df[:, 4].tolist(), target_df[:, 5].tolist()

        for imgpath, dt_bbox, dt_conf, TP_val, FP_val \
            in zip(target_test_image_list,
                   target_dt_bboxes_list,
                   target_dt_conf_list,
                   target_TP_list,
                   target_FP_list):

            target_sum_TP = target_sum_TP + TP_val
            target_sum_FP = target_sum_FP + FP_val

            target_precision = target_sum_TP / (target_sum_TP + target_sum_FP) if (target_sum_TP + target_sum_FP) != 0 else 0
            target_recall = target_sum_TP / class_gt_count[category_list[cnum]] if class_gt_count[category_list[cnum]] != 0 else 0

            target_PR_list.append([target_sum_TP, target_sum_FP, target_precision, target_recall])
        
        # print(target_PR_list)
        target_precision_list = np.array(target_PR_list)[:, 2] if target_PR_list != [] else []
        target_recall_list = np.array(target_PR_list)[:, 3] if target_PR_list != [] else []
        target_ap[category_list[cnum]] = AP(target_precision_list, target_recall_list)

    print()
    precision = round(precision, 4)
    recall = round(recall, 4)
    mAP = round(sum([x for x in target_ap.values()]) / len(target_ap), 4)
    f1_score = round(f1_score, 4)
    # make result csv part
    # print(total_TPFP_df)

    # ResultsPath 맨 뒤로 보내는 작업
    # colname = "results_path"
    # colindex = columns.index(colname)
    # reset_columns = columns[0:colindex] + columns[colindex + 1:] + columns[colindex:colindex + 1]
    # total_df = total_df[reset_columns]

    plt.clf()
    plt.plot(total_recall_list, total_precision_list, label="Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision x Recall Curve\nmAP: {str(mAP * 100)[:5]}%")
    plt.ylim([float(str(total_precision_list[-1])[0]) - 0.0225, 1.0225])
    plt.xlim([-0.05, 1.05])
    plt.grid(True)
    plt.legend(loc="best")

    # total_df.to_csv(csv_path, index=False)
    csv_filename = os.path.basename(csv_path)
    csv_savepath = os.path.split(csv_path)[0]
    plt_savename = f"{os.path.join(csv_savepath, csv_filename.split('.')[0])}-PR_Curve.jpg"
    plt.savefig(plt_savename)

    # make result csv part    
    # print(total_TPFP_df.columns)        # ['InputData', 'PredictedY_ClassName', 'PredictedY_BndBoxes', 'Confidence', 'TP', 'FP', 'SumTP', 'SumFP', 'Precision', 'Recall']
    recon_df = total_TPFP_df[['input_data', 'avg_iou', 'tp', 'fp']]
    TPFP_per_image = {x:{"avg_iou": list(), "tp": 0, "fp": 0} for x in test_image_list}
    for index, rowdata in recon_df.iterrows():
        TPFP_per_image[rowdata['input_data']]['avg_iou'].append(rowdata['avg_iou'])
        TPFP_per_image[rowdata['input_data']]['tp'] += rowdata['tp']
        TPFP_per_image[rowdata['input_data']]['fp'] += rowdata['fp']

    for key in TPFP_per_image.keys():
        TPFP_per_image[key]['avg_iou'] = get_average(TPFP_per_image[key]['avg_iou'])

    result_csv_list = list()
    _sum_TP, _sum_FP, _sum_FN = 0, 0, 0
    for data in zip(_test_image_list, result_image_list, gt_class_list, gt_bboxes_list, dt_class_list, dt_bboxes_list, dt_conf_list):
        filename = os.path.basename(data[0])
        # print(TPFP_per_image.keys())
        for _key in TPFP_per_image.keys():
            _filename = os.path.basename(_key)
            if _filename == filename:
                _avg_iou = TPFP_per_image[_key]['avg_iou']
                _TP = TPFP_per_image[_key]['tp']
                _FP = TPFP_per_image[_key]['fp']

        _FN = len(data[2]) - _TP
        _sum_TP += _TP
        _sum_FP += _FP
        _sum_FN += _FN

        _precision = _sum_TP / (_sum_TP + _sum_FP) if (_sum_TP + _sum_FP) != 0 else 0
        _recall = _sum_TP / total_gt_count if total_gt_count != 0 else 0

        result_csv_list.append(list(data) + [_avg_iou, _TP, _FP, _FN, _sum_TP, _sum_FP, _sum_FN, _precision, _recall])
    # print(result_csv_list)
    result_csv_df = pd.DataFrame(result_csv_list, columns=dataframe.columns.values.tolist() + ['avg_iou', 'tp', 'fp', 'fn', 'sum_tp', 'sum_fp', 'sum_fn', 'precision', 'recall'])

    colname = "results_path"
    colindex = result_csv_df.columns.values.tolist().index(colname)
    reset_columns = result_csv_df.columns.values.tolist()[0:colindex] \
        + result_csv_df.columns.values.tolist()[colindex + 1:] \
            + result_csv_df.columns.values.tolist()[colindex:colindex + 1]
    result_csv_df = result_csv_df[reset_columns]

    result_csv_df.to_csv(csv_path, index=False)

    print("\n ============== TOTAL METRIC SUMMARY ==============")
    table = [['Sum TP', str(sum_TP)], ['Sum FP', str(sum_FP)], ['Sum FN', str(_sum_FN)],
             ['Precision', str(precision)], ['Recall', str(recall)],
             ['F1 Score', str(f1_score)], ['mean AP', str(mAP)]]
    print(tabulate(table, tablefmt='grid'))
    for key, val in target_ap.items():
        print(f" -> Class [{key}] AP : {round(val, 4)}")


    print("\n ============== Metric END ==============\n")
    return [precision, recall, f1_score, float(mAP)]


def get_metric_for_segmentation(dataframe, csv_path, category_list=1):
    # is_dontcare_inTarget = False
    print("\n\n============== Semantic Segmentaion Metric START ==============")
    print(f"Now Date Time: {datetime.datetime.now()}")
    evaluator = SegmentationEvaluator(len(category_list))
    evaluator.reset()
    df_list = dataframe.values.tolist()

    columns = dataframe.columns.values.tolist() + ["pixel_acc", "pixel_acc_per_class", "mean_iou", "fw_iou"]
    vallist = []

    for elem in tqdm(df_list, desc=" Comparing between GT and prediction... "):
        input_data, result_images, gt, target = elem

        gt_image = Image.open(gt)
        gt_image = np.array(gt_image)
        target_image = Image.open(target)
        target_image = np.array(target_image)

        evaluator.add_batch(gt_image, target_image)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        vallist.append([input_data, result_images, gt, target, Acc, Acc_class, mIoU, FWIoU])

    sum_Acc = 0
    sum_Acc_class = 0
    sum_mIoU = 0
    sum_FWIoU = 0

    for val in vallist:
        sum_Acc = sum_Acc + val[-4]
        sum_Acc_class = sum_Acc_class + val[-3]
        sum_mIoU = sum_mIoU + val[-2]
        sum_FWIoU = sum_FWIoU + val[-1]

    sum_Acc = sum_Acc / len(vallist)
    sum_Acc_class = sum_Acc_class / len(vallist)
    sum_mIoU = sum_mIoU / len(vallist)
    sum_FWIoU = sum_FWIoU / len(vallist)

    table = [
        ["Pixel Accuracy", float(round(sum_Acc, 4))],
        ["Pixel Accuracy per Class", float(round(sum_Acc_class, 4))],
        ["Mean Intersection over Union(meanIoU)", float(round(sum_mIoU, 4))],
        ["Freq. Weighted Intersection over Union(FW IoU)", float(round(sum_FWIoU, 4))],
    ]

    print(tabulate(table, tablefmt="grid"))
    print("=============== Semantic Segmentaion Metric END ===============\n")

    answer = [
        float(round(sum_Acc, 4)),
        float(round(sum_Acc_class, 4)),
        float(round(sum_mIoU, 4)),
        float(round(sum_FWIoU, 4)),
    ]

    # job_id = csv_path.split("/")[-2]
    # csv_save_path = "/" + ("/").join(csv_path.split("/")[1:-1]) + "/"
    total_df = pd.DataFrame(vallist, columns=columns)
    # ResultsPath 맨 뒤로 보내는 작업
    colname = "results_path"
    colindex = columns.index(colname)
    reset_columns = columns[0:colindex] + columns[colindex + 1:] + columns[colindex:colindex + 1]
    total_df = total_df[reset_columns]

    total_df.to_csv(csv_path, index=False)

    return answer


def get_metric_for_segmentation(dataframe, csv_path, category_list=1):
    """This function returns segmentation metrics(pixel accuracy, meanIoU etc.)

    :param dataframe: A dataframe which has 
        [input_data path, composited image path, ground-truth path, result_colormap path]
    :param csv_path: Path of dataframe(.csv) file saved.
    :param category_list: A list of category.

    """
    # yt_prefix = path_join(prefix[0], prefix[1][0])
    # yp_prefix = path_join(prefix[0], prefix[1][1])

    # is_dontcare_inTarget = False
    print("\n\n============== Semantic Segmentaion Metric START ==============")
    print(f"Now Date Time: {datetime.datetime.now()}")
    evaluator = SegmentationEvaluator(len(category_list))
    evaluator.reset()
    df_list = dataframe.values.tolist()

    columns = dataframe.columns.values.tolist() + ["pixel_acc", "pixel_acc_per_class", "mean_iou", "fw_iou"]
    vallist = []

    for elem in tqdm(df_list, desc=" Comparing between GT and prediction... "):
        input_data, result_images, gt, target = elem

        gt_image = Image.open(gt)
        gt_image = np.array(gt_image)
        target_image = Image.open(target)
        target_image = np.array(target_image)

        evaluator.add_batch(gt_image, target_image)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        vallist.append([input_data, result_images, gt, target, Acc, Acc_class, mIoU, FWIoU])

    sum_Acc = 0
    sum_Acc_class = 0
    sum_mIoU = 0
    sum_FWIoU = 0

    for val in vallist:
        sum_Acc = sum_Acc + val[-4]
        sum_Acc_class = sum_Acc_class + val[-3]
        sum_mIoU = sum_mIoU + val[-2]
        sum_FWIoU = sum_FWIoU + val[-1]

    sum_Acc = sum_Acc / len(vallist)
    sum_Acc_class = sum_Acc_class / len(vallist)
    sum_mIoU = sum_mIoU / len(vallist)
    sum_FWIoU = sum_FWIoU / len(vallist)

    table = [
        ["Pixel Accuracy", float(round(sum_Acc, 4))],
        ["Pixel Accuracy per Class", float(round(sum_Acc_class, 4))],
        ["Mean Intersection over Union(meanIoU)", float(round(sum_mIoU, 4))],
        ["Freq. Weighted Intersection over Union(FW IoU)", float(round(sum_FWIoU, 4))],
    ]
    
    print(tabulate(table, tablefmt="grid"))
    print("=============== Semantic Segmentaion Metric END ===============\n")

    answer = [
        float(round(sum_Acc, 4)),
        float(round(sum_Acc_class, 4)),
        float(round(sum_mIoU, 4)),
        float(round(sum_FWIoU, 4)),
    ]

    # job_id = csv_path.split("/")[-2]
    # csv_save_path = "/" + ("/").join(csv_path.split("/")[1:-1]) + "/"
    total_df = pd.DataFrame(vallist, columns=columns)
    # ResultsPath 맨 뒤로 보내는 작업
    colname = "results_path"
    colindex = columns.index(colname)
    reset_columns = columns[0:colindex] + columns[colindex + 1:] + columns[colindex:colindex + 1]
    total_df = total_df[reset_columns]

    total_df.to_csv(csv_path, index=False)

    return answer
