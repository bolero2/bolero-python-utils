import os
import sys

PROJECT_HOME = os.getenv("PROJECT_HOME")
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
print("setting - PROJECT_HOME :", PROJECT_HOME)
print("setting - PYTHON_UTILS :", PYTHON_UTILS)

sys.path.append(PROJECT_HOME)
sys.path.append(PYTHON_UTILS)
os.environ['PROJECT_HOME'] = PROJECT_HOME
os.environ['PYTHON_UTILS'] = PYTHON_UTILS
