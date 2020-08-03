import os
import numpy as np
import cv2
from . import find_tools as ft

# noinspection PyArgumentList
hog = cv2.HOGDescriptor((32, 64), (16, 16), (8, 8), (8, 8), 9, 1, 4, 0, 0.2, 0, 64)


def find_right_lung_hog(image):
    hog.setSVMDetector(
        np.loadtxt(os.path.join(os.path.split(__file__)[0], 'right_lung_hog.np'), dtype=np.float32))
    found, w = hog.detectMultiScale(image)
    right_lung_rectangle = ft.find_max_rectangle(found)

    return right_lung_rectangle


def find_left_lung_hog(image):
    hog.setSVMDetector(np.loadtxt(os.path.join(os.path.split(__file__)[0], 'left_lung_hog.np'), dtype=np.float32))
    found, w = hog.detectMultiScale(image)
    left_lung_rectangle = ft.find_max_rectangle(found)

    return left_lung_rectangle
