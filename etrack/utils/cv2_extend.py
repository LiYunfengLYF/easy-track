import cv2
import numpy as np


def selectROI(winname: str, img: np.array, resize: [list, tuple]=None) -> list:
    """
    Description
        selectROI is an easy extension of cv2.selectROI
        input image is RGB rather BGR
    """

    if resize is not None:
        cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(winname, resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = cv2.selectROI(windowName=winname, img=img, showCrosshair=False)
    return bbox


def flip_img(image: np.array, horizontal: bool = False, vertical: bool = False) -> np.array:
    """
    Description
        flip_img is an easy extension of cv2.flip
        input image is RGB rather BGR
    """
    if horizontal:
        image = cv2.flip(image, 1)
    if vertical:
        image = cv2.flip(image, 0)
    return image


def clahe(image: np.array, clipLimit: int = 2.0, tileGridSize: tuple = (8, 8)) -> np.array:
    """
    Description
        clahe is an easy implement of CLAHE of cv2
        input image include (R,G,B) and (B,G,R)
    """
    B, G, R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_R = clahe.apply(R)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    return cv2.merge((clahe_B, clahe_G, clahe_R))
