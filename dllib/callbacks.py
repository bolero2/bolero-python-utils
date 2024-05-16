import json
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import re
import cv2

import logging
logger = logging.getLogger(__name__)


class CustomGetMetricCallback(object):
    """
    Metric 작성(in job-XXX-status.json)
    - runner_XXXXX.py에서 객체 생성 후(보통 torch가 될 것) fit 당시에 param으로 넘겨주는 형식.
    - 실제 모델 code에서 사용 당시
        epoch = 기록할 epoch 값
        metrics = [[train_metric], [valid_metric]]
        metrics_name = [[train_metric_name], [valid_metric_name]]

    example:
    my_getMetric = CustomGetMetricCallback(save_path=self.job.save_dir,
                                           _hash=self.job.hash)
    my_getmetric.save_status(epoch + 1,
                             metrics=[[round(mloss.cpu().numpy().tolist()[3], 4), round(mloss.cpu().numpy().tolist()[0], 4)],   # train metric
                                      [round(loss, 4), round(mAP, 4)]],                                                         # validation metric
                             metrics_name=[['total_loss', 'bbox_loss'],                                                         # train metric name
                                           ['total_loss', 'mAP']])                                                              # validation metric name
    """

    def __init__(self, save_path, _hash=""):
        self.metrics = []
        if os.path.splitext(os.path.basename(save_path))[1] != ".json":
            if _hash == "":
                from datetime import datetime
                nowdate = str(datetime.utcnow())
                filename = "losses_" + nowdate + ".json"

            else:
                filename = "losses_" + _hash + ".json"

            save_path = os.path.join(save_path, filename)
        self.save_path = os.path.abspath(save_path)

    def save_metrics(self, epoch, metrics=[], metrics_name=[]):
        # train_metrics_name = valid_metrics_name = self.metrics
        # Metrics 정제 과정
        if len(metrics) == 2 and (
            isinstance(metrics[0], list) and isinstance(metrics[1], list)
        ):
            train_metric = metrics[0]
            valid_metric = metrics[1]
        else:
            train_metric = metrics
            valid_metric = []

        train_metric = (
            [train_metric] if not isinstance(train_metric, list) else train_metric
        )
        valid_metric = (
            [valid_metric] if not isinstance(valid_metric, list) else valid_metric
        )

        # Metrics_name 정제 과정
        if len(metrics_name) == 2 and (
            isinstance(metrics_name[0], list) and isinstance(metrics_name[1], list)
        ):
            train_metric_name = metrics_name[0]
            valid_metric_name = metrics_name[1]
        else:
            train_metric_name = valid_metric_name = metrics_name

        train_metric_name = (
            [train_metric_name]
            if not isinstance(train_metric_name, list)
            else train_metric_name
        )
        valid_metric_name = (
            [valid_metric_name]
            if not isinstance(valid_metric_name, list)
            else valid_metric_name
        )

        for index, metric in enumerate(train_metric):
            if math.isnan(metric):
                train_metric[index] = "NaN"
        for index, metric in enumerate(valid_metric):
            if math.isnan(metric):
                valid_metric[index] = "NaN"

        # Dictionary type으로 변경 과정
        my_train_metric = {
            name: value for name, value in zip(train_metric_name, train_metric)
        }
        my_valid_metric = (
            {name: value for name, value in zip(valid_metric_name, valid_metric)}
            if len(valid_metric) != 0
            else {}
        )

        logger.info(
            "Metric Name is changed in [CustomGetMetricCallback function]\n: {}".format(
                (train_metric_name, valid_metric_name)
            )
        )

        # self.job.add_train_metric(epoch, my_train_metric, my_valid_metric)
        self.metrics.append({
            "epoch": epoch,
            "train_loss": my_train_metric,
            "valid_loss": my_valid_metric
        })

        print(
            " -> Added train metric > Epoch {} metrics [train: {}, valid: {}]".format(
                epoch, my_train_metric, my_valid_metric
            )
        )
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        with open(self.save_path, 'w') as f:
            json.dump({"metrics": self.metrics}, f, indent=2)

    def visualize_losses(self, logfile=""):
        logfile = self.save_path if logfile == "" else logfile
        
        from matplotlib import pyplot as plt
        with open(os.path.abspath(logfile), 'r') as f:
            jsondata = json.load(f)

        metric_data = jsondata['metrics']
        epochs = [x['epoch'] for x in metric_data]
        train_losses = [x['train_loss']['loss'] for x in metric_data]
        valid_losses = [x['valid_loss']['loss'] for x in metric_data]

        plt.figure(dpi=1200)
        plt.plot(epochs, train_losses, label='train')
        plt.plot(epochs, valid_losses, label='valid')
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(logfile), "losses_graph.png"))


class CustomSaveBestWeightCallback(object):
    """
    Best weight(best.pt, best.hdf5) 저장하기 위함
    - runner_XXXXX.py에서 객체 생성 후(보통 torch가 될 것) fit 당시에 param으로 넘겨주는 형식.
    - 실제 모델 code에서 사용 당시
        model = 저장할 model 객체
        model_metric = best epoch임을 판별하기 위해 비교 가능한 변수(val_loss, mAP 등) -> init val : inf / -inf
        compared = 해당 model_metric이 더 작아야 best 인지 더 커야 best 인지 판별하기 위함. ('less'/'more')
        lib = weight 저장 형식이 'KERAS'(=.hdf5) 인지 'TORCH'(=.pt) 인지 알려줌

    example:
    my_saveBestWeight = CustomSaveBestWeightCallback(save_path=self.job.save_dir,
                                                     _hash=self.job.hash,
                                                     lib='TORCH')
    my_savebestweight.save_best_weight(model=self, model_loss=avg_mAP, compared='more', lib='TORCH')
    """

    def __init__(self, save_path, _hash="", lib='TORCH'):
        self.valid_metric = False
        self.lib = lib.upper()
        ext = '.pt' if self.lib == 'TORCH' else '.hdf5'

        if _hash == "":
            from datetime import datetime
            nowdate = str(datetime.utcnow())
            filename = "best-weight_" + nowdate + ext
        else:
            filename = "best-weight_" + _hash + ext

        save_path = os.path.join(save_path, "weights") if save_path.split('/')[-1] != 'weights' else save_path
        save_path = os.path.join(save_path, filename)
        self.save_path = save_path

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def save_best_weight(self, model=None, model_metric=0.0, compared="less"):
        import torch

        # self.valid_metric initializing
        if self.valid_metric == False:
            if compared == "less":
                self.valid_metric = float("inf")
            elif compared == "more":
                self.valid_metric = float("-inf")

        assert model is not None, "Model is None state."
        self.model = model

        if compared == "less" and model_metric < self.valid_metric:
            logger.info(f"-> The model metric {model_metric} is less than the previous metric {self.valid_metric}")
            print("==========> model weight file(best.pt) is saved in {}\n".format(self.save_path))

            if self.lib == "KERAS":
                self.model.save_weights(self.save_path)
            elif self.lib == "TORCH":
                torch.save(self.model, self.save_path)
            self.valid_metric = model_metric

        elif compared == "more" and model_metric > self.valid_metric:
            logger.info(f"-> The model metric {model_metric} is greater than the previous metric {self.valid_metric}")
            print("==========> model weight file(best.pt) is saved in {}\n".format(self.save_path))
            
            if self.lib == "KERAS":
                self.model.save_weights(self.save_path)
            elif self.lib == "TORCH":
                torch.save(self.model, self.save_path)
            self.valid_metric = model_metric

        else:
            logger.info(f"The model metric {model_metric} is worse than the previous metric {self.valid_metric}\n")


class CustomEarlyStopping(object):
    """
    my_earlyStop = CustomEarlyStopping(patience=int(epochs * 0.1))
    """
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'-> EarlyStopping counter: now [{self.counter}] out of patience [{self.patience}] | Now Best Score : [{self.best_score}]')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0

if __name__ == "__main__":
    a = CustomGetMetricCallback(save_path='')

    a.visualize_losses("/home/bulgogi/bolero/projects/eb87a7b2da371ef3/losses_eb87a7b2da371ef3.json")
