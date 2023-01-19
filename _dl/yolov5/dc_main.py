import sys
import os
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
import benv
PROJECT_HOME = os.environ['PROJECT_HOME']

from get_model import get_model
from bcommon import get_hash
from random import shuffle
import pickle

from bdataset import DatasetParser
from bjob import Project
from datetime import datetime


if __name__ == "__main__":
    nowdate = str(datetime.utcnow()).replace(' ', '+')
    from argparse import Namespace

    job = Project()
    job.hash = get_hash()

    job.dataset_name = f'dataset/{str(job.hash)}.pkl'
    # job.dataset_name = "dataset/20221111-f516a08ef51af667.pkl"

    """
    dataset_parser = DatasetParser(imgpath="/home/bulgogi/bolero/dataset/det_dough/total/images",
                                   annotpath="/home/bulgogi/bolero/dataset/det_dough/total/annotations",
                                   pkl_path=job.dataset_name)
    """
    dataset_parser = DatasetParser(imgpath='/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/only_pepperoni/images',
                                   annotpath='/home/bulgogi/bolero/dataset/aistt_dataset/dcdataset/det_data/only_pepperoni/annotations',
                                   pkl_path=job.dataset_name)

    job.category_names = dataset_parser.get_category()
    job.num_categories = len(job.category_names)
    job.save_dir = savedir = job.project_fpath(_hash=job.hash)

    os.makedirs(savedir, exist_ok=True)

    print("\n - JOB HASH ID :", job.hash, "\n")

    job.yaml_file = os.path.abspath(os.path.join(os.getcwd(), "dc_yolo.yaml"))
    job.network_type = 'YOLOv5-Lite'
    job.target_size = [640, 640, 3]
    
    job.epochs = 500
    job.batch_size = 32

    job_data = job.__dict__
    job.save_job(job_data, os.path.join(savedir, f"{job.hash}.json"))

    model = get_model(job)
    dataset = dataset_parser.get_dataset()

    train_index = int(len(dataset[0]) * 0.8)
    valid_index = int(len(dataset[0]) * 0.9)

    train_data = [dataset[0][0:train_index], dataset[1][0:train_index], dataset[2][0:train_index], dataset[3][0:train_index]]
    valid_data = [dataset[0][train_index:valid_index], dataset[1][train_index:valid_index], dataset[2][train_index:valid_index], dataset[3][train_index:valid_index]]
    test_data = [dataset[0][valid_index:], dataset[1][valid_index:], dataset[2][valid_index:], dataset[3][valid_index:]]

    dataset_parser.save_dataset(train_data, os.path.join(savedir, f"train_{job.hash}.pkl"))
    dataset_parser.save_dataset(valid_data, os.path.join(savedir, f"valid_{job.hash}.pkl"))
    dataset_parser.save_dataset(test_data, os.path.join(savedir, f"test_{job.hash}.pkl"))

    model.fit(train_data[0], train_data[1], valid_data, epochs=job.epochs, batch_size=job.batch_size)
