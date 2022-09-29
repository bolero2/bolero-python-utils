import os
import json
import benv
from argparse import Namespace
benv.benv_setting()


class Project(object):
    def __init__(self):
        pass

    def project_fpath(self, _hash):
        project_path = os.getenv("PROJECT_HOME")
        return os.path.join(project_path, str(_hash))

    def weight_fpath(self, _hash):
        project_path = os.getenv("PROJECT_HOME")
        return os.path.join(project_path, _hash, "weights", f"best-weight_{_hash}.pt")

    def save_job(self, data:dict, path:str):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_job(self, path:str):
        from argparse import Namespace
        assert os.path.isfile(os.path.abspath(path)), "There isn't json file!" 

        job = Namespace()
        with open(os.path.abspath(path), 'r') as f:
            jsondata = json.load(f)

        for k, v in jsondata.items():
            job.__setattr__(k, v)

        return job
