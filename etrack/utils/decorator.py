import os
import sys


def no_print(func):
    def wrapper(*args, **kargs):
        sys.stdout = open(os.devnull, 'w')
        f = func(*args, **kargs)
        sys.stdout = sys.__stdout__
        return f

    return wrapper


def no_print_cv2(func):
    def wrapper(*args, **kargs):
        os.close(1)
        f = func(*args, **kargs)
        # os.dup2(2, 1)
        sys.stdout = sys.__stdout__
        return f

    return wrapper
