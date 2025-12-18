from typing import Literal

import torch
from torchvision import transforms


def get_transforms(
    image_size: int = 224,
    mode: Literal["none", "light", "realistic"] = "none",
):
    """
    mode:
      - none: resize + normalize
      - light: simple flips/rotations/color jitter (par défaut ancien comportement)
      - realistic: plus d'augmentations (flip, rotation, jitter, blur, légère noise)
    """
    ops = []
    if mode == "light":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    elif mode == "realistic":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        )
    ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())
    if mode == "realistic":
        # Bruit léger pour simuler capteur
        ops.append(transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)))
    ops.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    return transforms.Compose(ops)
