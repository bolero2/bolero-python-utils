from glob import glob
import os
import shutil as sh
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('-d', '--dir', help='target directory name', default='')
    parser.add_argument('-c', '--classes', help='classes list', type=lambda x:list(map(str, x.split(','))), default=list())    # onion,sweet_corn,sweet_potato_mousse
    args = parser.parse_args()

    print("Args :", args, "\n\n")

    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_rootpath = '/home/bulgogi/Desktop/sharing/230809/곽성호/labeled' if args.dir == '' else args.dir

    categories = ['bulgogi', 'green_pepper', 'mushroom', 'onion', 'roasted_onion_sauce'] if args.classes == [] else args.classes
    print(f"Categories : {categories} ({len(categories)})\n")
    
    print("[get_ingredient_dataset_from_total.py] dataset_rootpath :", dataset_rootpath)
    print("[get_ingredient_dataset_from_total.py] categories :", categories)
    
    labellist = glob(os.path.join(dataset_rootpath, "labels", "*.txt"))
    print(f"annotation file count : {len(labellist)}")

    for cat in categories:
        os.makedirs(os.path.join(dataset_rootpath, 'ingredients', cat), exist_ok=True)
        for name in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_rootpath, 'ingredients', cat, name), exist_ok=True)

    for annot in labellist:
        print("annot name :", annot)
        basename = os.path.basename(annot)
        save_name = {}

        with open(annot, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line[-1] != '\n':
                line = line + '\n'
            label_index = int(line.split(' ')[0])
            # print(label_index)
            # points = line.split(' ')[1]
            label_name = categories[label_index]

            temp_sentences_list = save_name.get(label_name, [])
            temp_sentences_list.append(line)
            save_name[label_name] = temp_sentences_list

        for target_name, target_point in save_name.items():

            sh.copy(os.path.join(dataset_rootpath, "images", basename.replace('.txt', '.jpg')), os.path.join(dataset_rootpath, "ingredients", target_name, "images"))

            for index, point in enumerate(target_point):
                point = point.split(' ')
                point[0] = '0'
                point = ' '.join(point)
                target_point[index] = point

            with open(os.path.join(dataset_rootpath, "ingredients", target_name, "labels", basename), 'w') as f:
                f.writelines(target_point)
