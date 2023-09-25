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
    elif isinstance(data, float):
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
    with open("/home/bulgogi/bolero/AisttGuideModels/DatasetProcessor/configs/labels.yaml") as f:
        categories = yaml.load(f, Loader=yaml.FullLoader)
    print(list(categories.values()))
    categories = list(categories.values())

    rootpath = '/home/bulgogi/Downloads/zuru-poc4/json-annotations'
    # jsonlist = sorted(glob(os.path.join(rootpath, "4ce6221e_pepperoni_20230724_043706.json")))
    jsonlist = sorted(glob(os.path.join(rootpath, "*.json")))

    for jsonfile in jsonlist:
        savename = os.path.join(rootpath, os.path.basename(jsonfile).replace('.json', '.txt'))

        sentences = []

        with open(jsonfile, 'r') as f:
            jsondata = json.load(f)

        shapes = jsondata.get('shapes', {})
        img_w = jsondata.get('imageWidth', 640)
        img_h = jsondata.get('imageHeight', 416)
        filename = jsondata.get('imagePath', '')

        for s in shapes:
            points = []
            labelname = s.get('label', '').replace(' ', '')
            labelindex = categories.index(labelname)
            _points = s.get('points', [])
            print(f"Now label name : {labelname}({labelindex})")

            points = parse_and_convert(_points, img_w, img_h, index=-1)     # list() 삭제 + 상대좌표로 변환하는 기능

            _sentence = f"{labelindex}"

            for p in points:
                try:
                    summation = sum(p)  # exception 위치
                    if isinstance(sum(p), float):
                        # print(summation)
                        _sentence += f" {p[0]} {p[1]}"

                except:
                    _sentence = f"{labelindex}"
                    print(f"Exception\n{traceback.print_exc()}")
                    for pp in p:
                        p_summation = sum(pp)
                        if isinstance(p_summation, float):
                            # print(p_summation)
                            _sentence += f" {pp[0]} {pp[1]}"
                    _sentence += "\n"

                    sentences.append(_sentence)
                    _sentence = ''

            if _sentence != '':
                _sentence += "\n"
                sentences.append(_sentence)
                _sentence = ''

            # points = str(points).replace('[', '').replace(']', '').replace(',', ' ')
            # points = points.replace('  ', ' ')

            # sentences.append(f"{labelindex} {points}\n")

            # print("Array :", points)
            # print("Label :", labelname)


            # if len(_points_arr.shape) == 1:
            #     for ss in _point

            # else:
            #     _points_arr[:, 0] = _points_arr[:, 0] / img_w
            #     _points_arr[:, 1] = _points_arr[:, 1] / img_h

        # print(sentences)

        with open(savename, 'w') as f:
            f.writelines(sentences)

