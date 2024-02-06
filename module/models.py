import torch
import torch.nn as nn
from torchvision import models

def get_model(name: str, **kwargs):
    if name == "resnet18":
        Resnet18(**kwargs)
    elif name == "custom":
        CustomModel(**kwargs)
    else:
        raise ValueError(f"{name} model is not found")


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_map = None

        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.fc = nn.Linear(512, num_classes)

        self.backbone.layer4.register_forward_hook(self.getFeatureMap)

    def getFeatureMap(self, module, inputs, outputs):
        self.feature_map = outputs

    def forward(self, x):
        x = self.backbone(x)
        return x, self.feature_map
    

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
        )
        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        feature_map = self.conv(x)
        x = self.gap(feature_map)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, feature_map