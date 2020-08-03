import torch
from .model import myNet
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os


def load_bone_model_pytorch():
    """
    Load and return the trained model.
    """
    model = myNet()
    model.load_state_dict(torch.load(os.path.join(os.path.split(__file__)[0], 'model_weights/net_weights.pth'),
                                     map_location=torch.device('cpu')))
    model.eval()
    return model


def get_inverted(image, model):
    # 3-channel uint8 images only
    img = np.copy(image)
    if len(np.shape(img)) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(Image.fromarray(img))
    with torch.no_grad() as tn:
        output = model(img[None, ...])
    _, pred = torch.max(output, 1)
    if np.asarray(pred)[0] != 1:
        print('Image inverted.')
        image = np.invert(image)
    return image
