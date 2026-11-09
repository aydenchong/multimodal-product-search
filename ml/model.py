import torch.nn as nn
from torchvision import models

def create_resnet18(num_classes=4):
    # Step 1: Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Step 2: Freeze pretrained backbone
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Step 3: Replace final classification layer
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model
