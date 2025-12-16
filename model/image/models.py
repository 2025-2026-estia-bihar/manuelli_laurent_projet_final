from torch import nn
from torchvision import models


def create_model(num_classes: int, dropout: float = 0.2, use_pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
