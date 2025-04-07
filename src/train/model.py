import torch
import torchvision
from torch import nn, optim
from torchvision.models.video import R3D_18_Weights

def load_new_model(num_classes, pretrained=True, device='cuda', freeze_params=True):
    """Load the pretrained model with the specified number of classes."""
    # Load pretrained model
    if pretrained:
        model = torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
    else:
        model = torchvision.models.video.r3d_18(weights=None)

    # Replace the classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Unfreeze all layers
    if freeze_params:
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Unfreeze the classifier
    for param in model.fc.parameters():
        param.requires_grad = True 

    model = model.to(device)

    return model