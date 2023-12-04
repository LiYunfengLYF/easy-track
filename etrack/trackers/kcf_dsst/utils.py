import numpy as np
import cv2


# ffttools
# 离散傅里叶变换、逆变换
def fftd(img, backwards=False, byRow=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    # return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!
    # DFT_INVERSE: 用一维或二维逆变换取代默认的正向变换,
    # DFT_SCALE: 缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用。
    # DFT_COMPLEX_OUTPUT: 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性

    if byRow:
        return cv2.dft(np.float32(img), flags=(cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT))
    else:
        return cv2.dft(np.float32(img),
                       flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))


# 实部图像
def real(img):
    return img[:, :, 0]


# 虚部图像
def imag(img):
    return img[:, :, 1]


# 两个复数，它们的积 (a+bi)(c+di)=(ac-bd)+(ad+bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


# 两个复数，它们相除 (a+bi)/(c+di)=(ac+bd)/(c*c+d*d) +((bc-ad)/(c*c+d*d))i
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def complexDivisionReal(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / b

    res[:, :, 0] = a[:, :, 0] * divisor
    res[:, :, 1] = a[:, :, 1] * divisor
    return res


# 可以将 FFT 输出中的直流分量移动到频谱的中央
def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))

    assert (img.ndim == 2)  # 断言，必须为真，否则抛出异常；ndim 为数组维数
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2  # shape[0] 为行，shape[1] 为列
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# recttools
# rect = {x, y, w, h}
# x 右边界
def x2(rect):
    return rect[0] + rect[2]


# y 下边界
def y2(rect):
    return rect[1] + rect[3]


# 限宽、高
def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


# 取超出来的边界
def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


# 经常需要空域或频域的滤波处理，在进入真正的处理程序前，需要考虑图像边界情况。
# 通常的处理方法是为图像增加一定的边缘，以适应 卷积核 在原图像边界的操作。
def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


def cutOutsize(num, limit):
    if num < 0:
        num = 0
    elif num > limit - 1:
        num = limit - 1
    return int(num)


def extractImage(img, cx, cy, patch_width, patch_height):
    xs_s = np.floor(cx) - np.floor(patch_width / 2)
    xs_s = cutOutsize(xs_s, img.shape[1])

    xs_e = np.floor(cx + patch_width - 1) - np.floor(patch_width / 2)
    xs_e = cutOutsize(xs_e, img.shape[1])

    ys_s = np.floor(cy) - np.floor(patch_height / 2)
    ys_s = cutOutsize(ys_s, img.shape[0])

    ys_e = np.floor(cy + patch_height - 1) - np.floor(patch_height / 2)
    ys_e = cutOutsize(ys_e, img.shape[0])

    return img[ys_s:ys_e, xs_s:xs_e]
