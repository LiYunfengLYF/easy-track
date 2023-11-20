import cv2


def selectROI(winname: str, img, resize=None):
    """
    Description
        selectROI is an extension of cv2.selectROI
        input image is RGB rather BGR

    Params:
        winname:    name
        img:        np.array

    return:
        bbox:       [x,y,w,h]

    """

    if resize is not None:
        cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(winname, resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = cv2.selectROI(windowName=winname, img=img, showCrosshair=False)
    return bbox


def flip_img(image, horizontal=False, vertical=False):
    if horizontal:
        image = cv2.flip(image, 1)
    if vertical:
        image = cv2.flip(image, 0)
    return image


def clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    B, G, R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    image = cv2.merge((clahe_B, clahe_G, clahe_R))
    return image
