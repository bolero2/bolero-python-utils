import sys
import os
import yaml
import glob
import json

from models.yolo import Model
from pprint import pprint
import logging

logger = logging.getLogger(__name__)

def network_type_dict(fullname=""):
    _dict = {
        "YOLOv5-Lite": "yolov5_lite",
        "YOLOv5-Lite-P4": "yolov5_lite_p4",
        "YOLOv5-Small": "yolov5s",
        "YOLOv5-Medium": "yolov5m",
        "YOLOv5-Large": "yolov5l",
        "YOLOv5-XLarge": "yolov5x"
    }
    
    return _dict[fullname] if fullname != "" else list(_dict.keys())


def get_model(job=None):
    yaml_file = job.yaml_file
    print(yaml_file)
    if os.path.isfile(yaml_file):
        with open(yaml_file) as f:
            setting = yaml.load(f, Loader=yaml.SafeLoader)
        assert len(setting), "either setting value must be specified"
    else:
        print("There isn't [setting_yaml] field in model.json file!")

    num_categories = job.num_categories
    category_names = job.category_names

    backbone = setting['backbone'] = job.network_type
    print(backbone)
    if backbone in ['yolov5_lite_dw', 'yolov5_lite_c3', 'yolov5_efficient_lite0']:
        print(f"[{backbone}] is not supported yet. It is overwritten [YOLOv5-Lite].")
        backbone = 'yolov5_lite'

    assert backbone in network_type_dict(), \
         f"Not supported backbone type [{backbone}] is entered."

    logger.info(f"Backbone Type : [{backbone}]")
    setting['model'] = setting['model'][network_type_dict(backbone)]

    setting["nc"] = setting["model"]["nc"] = num_categories
    setting["classes"] = setting["model"]["classes"] = category_names

    # print("\n[Setting.yaml -> Entire Setting Values]")
    # pprint(setting, indent=2)

    model = Model(cfg=setting, ch=3, nc=num_categories, job=job)

    return model


if __name__ == "__main__":
    from argparse import Namespace

    job = Namespace()

    job.network_type = 'YOLOv5-Lite'
    job.num_categories = 1
    job.category_names = ['face']
    job.save_dir = 'outputs'
    job.target_size = [416, 416]

    model = get_model(job)

    print(Namespace)
