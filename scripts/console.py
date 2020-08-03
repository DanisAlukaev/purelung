"""
Usage:
  augmentation --type=TYPE --file=PATH_FILE --result=PATH_RESULT [--indent=INDENT] [--equalize_out] [--auto_inverse]

Option:
  --type=TYPE            type of an augmentation, either 'crop' or 'suppression'
  --file=PATH_FILE       path to an input image
  --result=PATH_RESULT   path to an output image
  --indent=INDENT        indent from the bounding box of lungs [default: 5]
  --equalize_out         flag signalizing that the Histograms Equalization should be performed  [default: False]
  --auto_inverse         flag signalizing that the color inverse of an image should be performed if necessary  [default: False]

"""

import albumentations as albu
from purelung import augmentations as aug
import cv2
import docopt


def lung_crop(path_original, path_result, indent=5, auto_inverse=False):
    """
    The function crops an image with the given path and saves the result.

    :param path_original: a path to an input image.
    :param path_result: a path to store the result.
    :param indent: an indent from the bounding box of lungs.
    :param auto_inverse: a flag signalizing that the color inverse of an image should be performed if necessary.
    """
    image = cv2.imread(path_original, 0)
    crop = albu.Compose([aug.LungCrop(indent=indent, auto_inverse=auto_inverse)])
    image = crop(image=image)['image']
    cv2.imwrite(path_result, image)


def bone_suppression(path_original, path_result, equalize_out=False, auto_inverse=False):
    """
    The function performs ribs suppression in an image with given path.

    :param path_original: a path to an input image.
    :param path_result: a path to store the result.
    :param equalize_out: a flag signalizing that the Histograms Equalization should be performed.
    :param auto_inverse: a flag signalizing that the color inverse of an image should be performed if necessary.
    """
    image = cv2.imread(path_original, 0)
    suppress = albu.Compose([aug.RibSuppression(equalize_out=equalize_out, auto_inverse=auto_inverse)])
    image = suppress(image=image)['image']
    cv2.imwrite(path_result, image)


def main():
    args = docopt.docopt(__doc__)
    aug_type = args['--type']
    original = args['--file']
    result = args['--result']
    indent = args['--indent']
    auto_inverse = args['--auto_inverse']
    equalize_out = args['--equalize_out']

    auto_inverse = True if auto_inverse else False
    if aug_type == 'crop':
        indent = int(indent) if indent else 5
        lung_crop(original, result, indent=indent, auto_inverse=auto_inverse)
    elif aug_type == 'suppression':
        equalize_out = True if equalize_out else False
        bone_suppression(original, result, equalize_out=equalize_out, auto_inverse=auto_inverse)


if __name__ == '__main__':
    main()
