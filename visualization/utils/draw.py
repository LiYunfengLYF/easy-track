import cv2


def draw_box(image, box, color, thickness):
    """
    :param image: cv2 image
    :param box: [x, y, w, h]
    :param color: (r,g,b)
    :param thickness: int
    :return:
    """

    tlx, tly = int(box[0]), int(box[1])
    brx, bry = int(box[0] + box[2]), int(box[1] + box[3])
    image = cv2.rectangle(image, (tlx, tly), (brx, bry), color=color, thickness=thickness)
