import os
import json


def save_job(data:dict, path:str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_job(path:str):
    from argparse import Namespace
    assert os.path.isfile(os.path.abspath(path)), "There isn't json file!" 

    job = Namespace()
    with open(os.path.abspath(path), 'r') as f:
        jsondata = json.load(f)

    for k, v in jsondata.items():
        job.__setattr__(k, v)
        # eval(f"job.{k}") = v


    print(job)

    return job
