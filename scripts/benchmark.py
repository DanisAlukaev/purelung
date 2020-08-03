from timeit import default_timer as timer
import cv2
from purelung import augmentations as aug
import os

augmentation = aug.LungCrop()
# augmentation = aug.RibSuppression()

start = timer()
for i in range(2000):
    path = os.path.join('home', 'intern', 'dataset_png', str(i + 1) + ".png")
    image = cv2.imread(path, 0)
    augmentation(image=image)
end = timer()
time = end - start
print(time)
