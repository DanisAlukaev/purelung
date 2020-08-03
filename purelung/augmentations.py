import albumentations as albu
import cv2
import numpy as np
from .models.segmentation import segmentation
from .models.suppression import bone_suppression_pt as suppression_pt
from .models.lungs_finder import find_tools
from .models.inverse import inverted as inverse
from enum import Enum


class SegType(Enum):
    LUNGS = 0
    HEART = 1
    CLAVICLES = 2


class LungCrop:
    """
    Class used for context-aware image crop in the Chest X-ray images.

    targets: image, mask, bbox, key points.
    image types: uint8.
    """

    def __init__(self, indent=5, auto_inverse=False):
        """
        Set an indent from the bounding box of lungs.

        :param indent: an indent from the bounding box of lungs.
        :param auto_inverse: a flag signalizing that the color inverse of an image should be performed if necessary.
        """
        if not isinstance(indent, int):
            raise TypeError("Expected type 'int', got " + type(indent).__name__ + ".")
        self.indent = indent
        self.auto_inverse = auto_inverse

    def __call__(self, image, bbox=None, mask=None, keypoints=None, force_apply=False, *args, **kwargs):
        """
        Perform an image crop based on the location of lungs.

        :param image: an image to be cropped.
        :param bbox: a bounding box in pascal voc format, e.g. [x_min, y_min, x_max, y_max].
        :param mask: a mask for an input image.
        :param keypoints: a list of points in 'xy' format, e.g. [(x, y), ...].
        :return: a cropped image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected type 'numpy.ndarray', got " + type(image).__name__ + ".")

        if self.auto_inverse:
            image = inverse.get_inverted(image, inverse.load_bone_model_pytorch())

        iw, ih = np.shape(image)
        temp_image = image
        params = find_tools.get_lungs(temp_image)
        if params is not None:
            x, y, w, h = params
            # get the leftmost top point, width and height of a bounding rectangle
            height, width = image.shape
            if w > iw * 0.59 and h > ih * 0.59:
                # check whether the bounding rectangle extends beyond the image
                if x - self.indent < 0 or y - self.indent < 0 or x + w + self.indent > width or y + h + self.indent > height:
                    print('Values for crop should be non negative and equal or smaller than image size.')
                    # compute an optimal indent from bounding rectangle
                    self.indent -= max((lambda: 0 if x > self.indent else self.indent - x)(),
                                       (lambda: 0 if y > self.indent else self.indent - y)(),
                                       (
                                           lambda: 0 if x + w + self.indent < width else x + w + self.indent - width)(),
                                       (
                                           lambda: 0 if y + h + self.indent < height else y + h + self.indent - height)())
                    print('Indent reduced to ', self.indent, '.', sep='')

                # compute minimum upper left x y coordinates and maximum lower right x y coordinates.
                x_min = x - self.indent
                y_min = y - self.indent
                x_max = x + w + self.indent
                y_max = y + h + self.indent
                # compose an augmentation function
                crop_augmentation = albu.Crop(x_min=x_min, y_min=y_min, x_max=x_max,
                                              y_max=y_max, always_apply=False, p=1.0)
                # crop an input image
                cropped = crop_augmentation(image=temp_image)
                cropped['mask'] = None
                cropped['bbox'] = None
                cropped['keypoints'] = None

                # update the location of the bounding box
                if bbox:
                    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
                    bbox_x_min -= x_min
                    bbox_y_min -= y_min
                    bbox_x_max -= (abs(bbox_x_max - x_max) + x_min)
                    bbox_y_max -= (abs(bbox_y_max - y_max) + y_min)
                    bbox_x_min = max(0, bbox_x_min)
                    bbox_y_min = max(0, bbox_y_min)
                    bbox_x_max = max(0, bbox_x_max)
                    bbox_y_max = max(0, bbox_y_max)
                    cropped['bbox'] = [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]

                # crop mask
                if mask is not None:
                    cropped['mask'] = crop_augmentation(image=mask)['image']
                else:
                    cropped['mask'] = None

                # update coordinates of the input key points
                if keypoints:
                    new_keypoints = []
                    for point in keypoints:
                        x_coord, y_coord = point
                        x_coord -= x_min
                        y_coord -= y_min
                        if x_coord > 0 and y_coord > 0:
                            new_keypoints.append((x_coord, y_coord))
                    cropped['keypoints'] = new_keypoints
            else:
                cropped = {'image': image, 'bbox': bbox, 'mask': mask, 'keypoints': keypoints}
        else:
            # the image remains unchanged
            print("Lungs are not found on image.")
            cropped = {'image': image, 'bbox': bbox, 'mask': mask, 'keypoints': keypoints}
        return cropped


class RibSuppression:
    """
    Class used for suppressing bone shadows in the Chest X-ray images.

    targets: image, mask, bbox, key points.
    image types: uint8.
    """

    def __init__(self, equalize_out=False, auto_inverse=False):
        """
        Set a Histograms Equalization flag.

        :param equalize_out: a flag signalizing that the Histograms Equalization should be performed.
        :param auto_inverse: a flag signalizing that the color inverse of an image should be performed if necessary.
        """
        if not isinstance(equalize_out, bool):
            raise TypeError("Expected type 'bool', got " + type(equalize_out).__name__ + ".")

        self.equalize_out = equalize_out
        self.auto_inverse = auto_inverse
        # get the trained model from unet_resnet1024.pth file
        self.model_suppression = suppression_pt.load_bone_model_pytorch()

    def __call__(self, image, bbox=None, mask=None, keypoints=None, force_apply=False, *args, **kwargs):
        """
        Perform the ribs suppression in chest X-ray scans.

        :param image: an image that will be processed.
        :param bbox: a bounding box in pascal voc format, e.g. [x_min, y_min, x_max, y_max].
        :param mask: a mask for an input image.
        :param keypoints: a list of points in 'xy' format, e.g. [(x, y), ...].
        :return: a suppressed image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected type 'numpy.ndarray', got " + type(image).__name__ + ".")

        if self.auto_inverse:
            # get inverse image if necessary
            image = inverse.get_inverted(image, inverse.load_bone_model_pytorch())

        if len(np.shape(image)) > 2:
            # convert image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get an image with suppressed bones
        suppressed = suppression_pt.get_suppressed_image(image, self.model_suppression, equalize_out=self.equalize_out)
        return {'image': suppressed, 'bbox': bbox, 'mask': mask, 'keypoints': keypoints}


class Segmentation:
    """
    Class used for segmentation of the Chest X-ray images.

    targets: image, mask, bbox, key points.
    image types: uint8.
    """

    def __init__(self, auto_inverse=False, type=SegType.LUNGS):
        """
        Set an auto_inverse parameter.
        :param auto_inverse: a flag signalizing that the color inverse of an image should be performed if necessary.
        :param type: segmentation type, either SegType.LUNGS or SegType.HEART or SegType.CLAVICLES.
        """
        self.auto_inverse = auto_inverse
        self.type = type
        # create a new instance of the segmentation model
        self.segmentation_model = segmentation.Segmentation()
        # get a computational graph for this model
        self.segmentation_graph = self.segmentation_model.graph

    def __call__(self, image, bbox=None, mask=None, keypoints=None, force_apply=False, *args, **kwargs):
        """
        Perform the lung segmentation in chest X-ray scans.

        :param image: an image that will be processed.
        :param bbox: a bounding box in pascal voc format, e.g. [x_min, y_min, x_max, y_max].
        :param mask: a mask for an input image.
        :param keypoints: a list of points in 'xy' format, e.g. [(x, y), ...].
        :return: a segmentation mask.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected type 'numpy.ndarray', got " + type(image).__name__ + ".")

        if self.auto_inverse:
            image = inverse.get_inverted(image, inverse.load_bone_model_pytorch())

        iw, ih = np.shape(image)
        with self.segmentation_graph.as_default():
            # downscale image to (512, 512)
            temp_image = cv2.resize(image, (512, 512))
            # check whether an image is already in the grayscale
            if len(np.shape(temp_image)) > 2:
                # convert image to grayscale
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
            # perform the Histogram Equalization over the image
            temp_image = cv2.equalizeHist(temp_image)
            # change the color space of an image after Histogram Equalization
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
            seg_mask = []
            if self.type == SegType.LUNGS:
                # get the binary mask of lungs
                seg_mask = self.segmentation_model.predict_lungs(temp_image) * 255
            elif self.type == SegType.HEART:
                # get the binary mask of heart
                seg_mask = self.segmentation_model.predict_heart(temp_image) * 255
            elif self.type == SegType.CLAVICLES:
                # get the binary mask of clavicles
                seg_mask = self.segmentation_model.predict_clavicles(temp_image) * 255
            # resize mask from (512, 512) to original shape
            seg_mask = cv2.resize(seg_mask, (ih, iw))
            return {'image': seg_mask, 'bbox': bbox, 'mask': mask, 'keypoints': keypoints}


class Positive:
    """
    Class used for auto inverse of the Chest X-ray images.

    targets: image, mask, bbox, key points.
    image types: uint8.
    """

    def __init__(self):
        pass

    def __call__(self, image, bbox=None, mask=None, keypoints=None, force_apply=False, *args, **kwargs):
        positive = inverse.get_inverted(image, inverse.load_bone_model_pytorch())
        return {'image': positive, 'bbox': bbox, 'mask': mask, 'keypoints': keypoints}
