import json
import os
from glob import glob
import yaml
import traceback
import numpy as np


def parse_and_convert(data, w, h, index):
    if isinstance(data, list):
        data = list(data)
        return [parse_and_convert(item, w, h, i) for i, item in enumerate(data)]
    elif isinstance(data, float) or isinstance(data, int):
        if index == 0:
            data = float(str(data / w)[0:6])
        elif index == 1:
            data = float(str(data / h)[0:6])
        return data


def apply_function_to_nested_list(lst, func):
    result = []
    for item in lst:
        # print("Item :", item)
        if isinstance(item, list):
            # print("here is list")
            result.append(apply_function_to_nested_list(item, func))
        else:
            result.append(item)
            
    return result


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/bolero/dataset/2023aimmo/에이모_작업본/'
    jsonlist = glob(os.path.join(dataset_rootpath, "**", "*.json"), recursive=True)

    for jsonfile in jsonlist:
        savename = os.path.join(os.path.dirname(jsonfile), os.path.basename(jsonfile).replace('.json', '.txt'))
        savename = savename.replace(".jpg", "")
        sentences = []

        with open(jsonfile, 'r') as f:
            jsondata = json.load(f)

        _categories = jsondata.get('categories', [])
        categories = [x['name'] for x in _categories]
        print(categories)

        shapes = jsondata.get('shapes', []) 
        img_w = jsondata.get('width', 640)
        img_h = jsondata.get('height', 640)
        filename = jsondata.get('filename', '')

        for s in shapes:
            points = []
            labelname = s.get('label', '').replace(' ', '')
            labelindex = categories.index(labelname)
            _points = s.get('points')
            print(f"Now label name : {labelname}({labelindex})")
            points = []
            print(_points)
            for p in _points:
                points.append(str(p[0] / img_w))
                points.append(str(p[1] / img_h))

            points = ' '.join(points)

            _sentence = f"{labelindex} {points}"

            _sentence += "\n"
            sentences.append(_sentence)
            _sentence = ''

        with open(savename, 'w') as f:
            f.writelines(sentences)

