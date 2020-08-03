import os
import cv2
from . import find_tools as ft


def find_right_lung_haar(image):
    right_lung = cv2.CascadeClassifier(os.path.join(os.path.split(__file__)[0], 'right_lung_haar.xml'))
    found = right_lung.detectMultiScale(image, 1.8, 5)
    right_lung_rectangle = ft.find_max_rectangle(found)

    return right_lung_rectangle


def find_left_lung_haar(image):
    left_lung = cv2.CascadeClassifier(os.path.join(os.path.split(__file__)[0], 'left_lung_haar.xml'))
    found = left_lung.detectMultiScale(image, 1.8, 5)
    left_lung_rectangle = ft.find_max_rectangle(found)

    return left_lung_rectangle
