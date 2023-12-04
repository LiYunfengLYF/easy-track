import cv2
import numpy as np


def draw_box(image: np.array, box: list, color: tuple, thickness: int) -> np.array:
    """
    Description
        draw a bounding box on image
    """
    tlx, tly = int(box[0]), int(box[1])
    brx, bry = int(box[0] + box[2]), int(box[1] + box[3])
    image = cv2.rectangle(image, (tlx, tly), (brx, bry), color=color, thickness=thickness)
    return image


def draw_txt(image: np.array, text: str, org: [list, tuple] = (20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
             fontScale: int = 2, color: tuple = (0, 0, 255), thickness: int = 3) -> np.array:
    """
    Description
        draw a txt on image
        an easy implement of cv2.putText
    """
    return cv2.putText(image, text, org, fontFace, fontScale, color, thickness)
