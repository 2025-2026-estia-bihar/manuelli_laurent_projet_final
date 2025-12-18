from typing import Optional

import torch
from torch import nn
from torchvision import models


def _resolve_weights(model_name: str, pretrained: bool):
    if not pretrained:
        return None
    weight_enum = getattr(models, f"{model_name.upper()}_Weights", None)
    if weight_enum is None:
        return None
    return weight_enum.DEFAULT


def create_resnet(
    model_name: str = "resnet18",
    num_classes: int = 4,
    dropout: float = 0.2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    model_fn = getattr(models, model_name)
    weights = _resolve_weights(model_name, pretrained)
    kwargs = {"weights": weights} if weights is not None else {"pretrained": pretrained}
    backbone = model_fn(**kwargs)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    if freeze_backbone:
        for name, param in backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    return backbone
