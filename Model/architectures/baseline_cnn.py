import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, dropout=dropout / 2),
            conv_block(32, 64, dropout=dropout / 2),
            conv_block(64, 128, dropout=dropout / 2),
            conv_block(128, 256, dropout=dropout / 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_baseline_cnn(num_classes: int = 4, dropout: float = 0.3) -> nn.Module:
    return BaselineCNN(num_classes=num_classes, dropout=dropout)
