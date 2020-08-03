import torch
from .unet_resnet import myResnetUnet
import cv2
import numpy as np
import os


def load_bone_model_pytorch():
    """
    Load and return the trained model.
    """
    model = myResnetUnet(1, 1, 'relu')
    model.load_state_dict(torch.load(os.path.join(os.path.split(__file__)[0], 'model_weights/unet_resnet1024.pth'),
                                     map_location=torch.device('cpu')))
    model.eval()
    return model


def get_suppressed_image(image, model, equalize_out=False):
    """
    Perform bone suppression in an input image.

    :param image: an image, in which ribs should be suppressed.
    :param model: a pre-trained model for bone suppression.
    :param equalize_out: a flag signalizing that the Histograms Equalization should be performed.
    :return: an image, in which bone should be suppressed.
    """
    # create an array to store modified image
    img = np.copy(image)
    img_shape = img.shape
    if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
        new_shape = ((img.shape[0] // 8 + 1) * 8, (img.shape[1] // 8 + 1) * 8)
        img_temp = np.zeros(new_shape, dtype=np.uint8)
        img_temp[:img_shape[0], :img_shape[1]] = img
        img = img_temp
    img_torch = torch.from_numpy(img).unsqueeze(0).unsqueeze(1).type(torch.float32)
    # get the result of suppression
    with torch.no_grad() as tn:
        pred = model(img_torch)
    res = pred[0, 0, :img_shape[0], :img_shape[1]].numpy()
    res = np.clip(np.round(res), 0, 255).astype(np.uint8)
    # perform the Histogram Equalization over the image
    if equalize_out:
        res = cv2.equalizeHist(res.astype(np.uint8))
    return res
