from glob import glob
import os
import shutil as sh


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/Desktop/newpizza/train'
    categories = ['basil_oil', 'marinated_tomato', 'tomato_sauce']
    labellist = glob(os.path.join(dataset_rootpath, "labels", "*.txt"))

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
            print(label_index)
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
