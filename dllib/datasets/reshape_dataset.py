import os
from glob import glob
import cv2


def reshape_416640(img):
    """
    img: np.ndarray type, img = cv2.imread("a.jpg", cv2.IMREAD_COLOR)
    """
    ih, iw, ic = img.shape
    new_iw = 416 if abs(416 - iw) < abs(640 - iw) else 640
    new_ih = 416 if abs(416 - ih) < abs(640 - ih) else 640

    img_reshaped = cv2.resize(img, (new_iw, new_ih))
    print(f" ⭕️ Convert ({iw}, {ih}) 👉🏻 ({new_iw}, {new_ih})")
    
    return img_reshaped


if __name__ == "__main__":
    dataset_rootpath = '/home/bulgogi/Desktop/sharing/_finished/20230629이전 요청 레이블링(gorgonzola_pizza)'
    imglist = glob(os.path.join(dataset_rootpath, "**", "*.jpg"), recursive=True)

    for i in imglist:
        img = cv2.imread(i, cv2.IMREAD_COLOR)
        img = reshape_416640(img)
        cv2.imwrite(i, img)
