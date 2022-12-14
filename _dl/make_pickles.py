import pickle
import os
import sys
from glob import glob


trainlist = glob(os.path.join("original", "train", 'images', '*.jpg'))
validlist = glob(os.path.join("original", "valid", 'images', '*.jpg'))
testlist = glob(os.path.join("original", "test", 'images', '*.jpg'))

trainpack, validpack, testpack = [], [], []

for elem in trainlist:
    imagename = os.path.abspath(elem)
    annotname = os.path.join(os.path.dirname(os.path.abspath(elem)).replace('/images', '/annotations'), os.path.splitext(os.path.basename(elem))[0] + ".png")
    trainpack.append({imagename: annotname})

for elem in validlist:
    imagename = os.path.abspath(elem)
    annotname = os.path.join(os.path.dirname(os.path.abspath(elem)).replace('/images', '/annotations'), os.path.splitext(os.path.basename(elem))[0] + ".png")
    validpack.append({imagename: annotname})

for elem in testlist:
    imagename = os.path.abspath(elem)
    annotname = os.path.join(os.path.dirname(os.path.abspath(elem)).replace('/images', '/annotations'), os.path.splitext(os.path.basename(elem))[0] + ".png")
    testpack.append({imagename: annotname})

# (dc) dataset state save pickle format
pklpath = os.path.join("data", "pickles", "dataset_with_original") 
if not os.path.isdir(pklpath):
    os.makedirs(pklpath, exist_ok=True)

train_pkl_path = os.path.join(pklpath, "train.pkl")
valid_pkl_path = os.path.join(pklpath, "valid.pkl")
test_pkl_path = os.path.join(pklpath, "test.pkl")

with open(train_pkl_path, "wb") as pkl_file:
    pickle.dump(trainpack, pkl_file)
with open(valid_pkl_path, "wb") as pkl_file:
    pickle.dump(validpack, pkl_file)
with open(test_pkl_path, "wb") as pkl_file:
    pickle.dump(testpack, pkl_file)

