import torch
import torch.nn as nn
from torchvision import models

def create_model(arch, hidden_units):

    # Load pretrained model
    if arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    return model