from glob import glob
import os
from tqdm import tqdm
import cv2
import pickle


class DatasetParser:
    def __init__(self, imgpath, annotpath, category_file="", pkl_path=""):
        self.imgpath = imgpath
        self.annotpath = annotpath
        if pkl_path == "":
            self.pkl_path = os.path.abspath(pkl_path) + ".pkl"
        else:
            self.pkl_path = pkl_path

        common_prefix = os.path.commonprefix([self.imgpath, self.annotpath])
        self.category_file = category_file if category_file != "" else os.path.join(common_prefix, "class.txt")
        print(" - dataset pickle path :", self.pkl_path)

    def get_dataset(self):
        """
        Input: image path, annotation path(txt)
        Return: [image list, annotation list, width, height]

            imagelist -> abs path
            annotation list -> ccwh type, [label, center_x, center_y, width, height]
            widths -> group of image width 
            heights -> group of image height
        """
        dataset = None

        if self.pkl_path != "" and os.path.isfile(self.pkl_path):
            dataset = self.load_dataset(self.pkl_path)

        else:
            dataset = []
            images, annotations, widths, heights = [], [], [], []

            imglist = glob(os.path.join(self.imgpath, "*.png"))
            assert len(imglist) > 0, "There isn't image files! Please check paths."

            for i, imgname in tqdm(enumerate(imglist), total=len(imglist), desc='Parsing Dataset ... '):
                assert os.path.isfile(imgname)

                img = cv2.imread(imgname)
                ih, iw, ic = img.shape

                images.append(imgname)
                widths.append(iw)
                heights.append(ih)

                basename = os.path.basename(imgname)
                _basename, _ext = os.path.splitext(basename)
                annotname = os.path.join(self.annotpath, f"{_basename}.txt")
                assert os.path.isfile(annotname)

                annotfile = open(annotname, "r")
                annotlines = annotfile.readlines()
                temp_annots = []

                for al in annotlines:
                    annotval = list(map(float, al.split(' ')))
                    annotval[0] = int(annotval[0])
                    temp_annots.append(annotval)
                annotations.append(temp_annots)

            assert len(images) == len(annotations) == len(widths) == len(heights), "Dataset parsing was wrong!"
            dataset = [images, annotations, widths, heights]

            self.save_dataset(dataset, self.pkl_path)
            
        return dataset 

    def get_category(self):
        assert os.path.isfile(self.category_file), "There isn't category information! Enter correct category file or check class.txt file."

        with open(self.category_file, 'r') as f:
            categories = f.readlines()

        for idx in range(len(categories)):
            categories[idx] = categories[idx].replace('\n', '')

        return categories

    def save_dataset(self, dataset: list, pkl_path: str):
        print("Save dataset to pickle format.")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(dataset, pkl_file)

    @staticmethod
    def load_dataset(pkl_path: str):
        print(f"\n * Load dataset from pickle format : {os.path.basename(pkl_path)}")
        dataset = None

        with open(pkl_path, "rb") as pkl_file:
            dataset = pickle.load(pkl_file)

        print(f"    - {os.path.basename(pkl_path)} Dataset Size :", len(dataset), "\n")
        return dataset


if __name__ == "__main__":
    # dataset = get_dataset("/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/images", "/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/annotations") 
    dataset = get_dataset("/home/bulgogi/bolero/dataset/small_catdog_Detection_dataset/images", "/home/bulgogi/bolero/dataset/small_catdog_Detection_dataset/annotations")

    print(dataset)
    print(len(dataset))
    for i in range(len(dataset)):
        print(len(dataset[i]))

    """
    if os.path.isfile(f"dataset/{job.dataset_name}"):
        with open(f"dataset/{job.dataset_name}", "rb") as pck_dataset:
            dataset = pickle.load(pck_dataset)

    else:
        dataset = get_dataset("/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/images",
                            "/home/bulgogi/bolero/dataset/WiderFaceDetectionDataset/WIDER_total/annotations")
        os.makedirs("dataset", exist_ok=True)
        with open(f"dataset/{job.dataset_name}", "wb") as pck_dataset:
            pickle.dump(dataset, pck_dataset)
    """
