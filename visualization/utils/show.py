import cv2


def imshow(winname, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
    cv2.imshow(winname, image)
    cv2.waitKey(0)


def close_cv2_window(seq_name):
    cv2.destroyWindow(seq_name)
