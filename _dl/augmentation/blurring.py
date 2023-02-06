import cv2

def gaussian_blur(img):
    sigma = 7
    dst = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return dst
