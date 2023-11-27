import os
import sys

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
print("setting - PYTHON_UTILS :", PYTHON_UTILS)
sys.path.append(PYTHON_UTILS)
os.environ['PYTHON_UTILS'] = PYTHON_UTILS


def benv_setting():
    PYTHON_UTILS = os.getenv("PYTHON_UTILS")
    print("setting - PYTHON_UTILS :", PYTHON_UTILS)
    sys.path.append(PYTHON_UTILS)
    os.environ['PYTHON_UTILS'] = PYTHON_UTILS
