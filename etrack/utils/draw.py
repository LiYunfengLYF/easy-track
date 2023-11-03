import cv2


def draw_box(image, box, color, thickness):
    """
    Description
        draw a bounding box on image

    Params:
        image:      np.array
        box:        [x, y, w, h]
        color:      bounding box color
        thickness:  bounding box thickness

    Return:
        image:      np.array

    """
    tlx, tly = int(box[0]), int(box[1])
    brx, bry = int(box[0] + box[2]), int(box[1] + box[3])
    image = cv2.rectangle(image, (tlx, tly), (brx, bry), color=color, thickness=thickness)
    return image
