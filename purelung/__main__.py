import purelung.augmentations as aug
import cv2
import numpy as np


def main():
    image = cv2.imread(r'C:\Users\pc\Desktop\original.png', 0)
    image = np.invert(image)
    augmentation = aug.Segmentation(type=aug.SegType.CLAVICLES)
    augmented = augmentation(image=image)['image']
    cv2.imshow('augmented', augmented)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
