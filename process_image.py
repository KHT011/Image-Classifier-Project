import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    # Convert color channel values
    np_image = np.array(preprocess(image)) / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose
    np_image = np_image.transpose(2, 0, 1)

    return torch.Tensor(np_image)