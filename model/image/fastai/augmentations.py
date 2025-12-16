from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
from fastai.vision.all import PILImage, Transform


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 0.25) -> np.ndarray:
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 1)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.08, salt_vs_pepper: float = 0.5) -> np.ndarray:
    noisy = np.copy(image)
    h, w, c = image.shape
    num_salt = int(np.ceil(amount * h * w * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * h * w * (1.0 - salt_vs_pepper)))

    coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
    for ch in range(c):
        noisy[coords[0], coords[1], ch] = 1

    coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
    for ch in range(c):
        noisy[coords[0], coords[1], ch] = 0

    return np.clip(noisy, 0, 1)


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    return np.clip(np.random.poisson(image * 255) / 255.0, 0, 1)


def add_speckle_noise(image: np.ndarray, scale: float = 0.35) -> np.ndarray:
    noise = np.random.randn(*image.shape) * scale
    return np.clip(image + image * noise, 0, 1)


@dataclass
class NoiseConfig:
    same_across_channels: bool = True
    gaussian_std: float = 0.25
    saltpepper_amount: float = 0.08
    speckle_scale: float = 0.35


class AddRandomNoise(Transform):
    """
    Random noise augmentation applied only on train split.
    """
    split_idx = 0

    def __init__(self, cfg: NoiseConfig = NoiseConfig()):
        self.cfg = cfg
        self.noise_funcs: List[Tuple[str, Callable[..., np.ndarray]]] = [
            ("Gaussian", partial(add_gaussian_noise, std=self.cfg.gaussian_std)),
            ("SaltPepper", partial(add_salt_pepper_noise, amount=self.cfg.saltpepper_amount)),
            ("Poisson", add_poisson_noise),
            ("Speckle", partial(add_speckle_noise, scale=self.cfg.speckle_scale)),
        ]

    def encodes(self, img: PILImage) -> PILImage:
        img_np = np.array(img).astype(np.float32) / 255.0
        _, func = random.choice(self.noise_funcs)

        if self.cfg.same_across_channels:
            noisy_np = func(img_np)
        else:
            # apply per channel
            noisy_np = np.stack([func(img_np[..., c]) for c in range(3)], axis=-1)

        noisy_uint8 = (noisy_np * 255).astype(np.uint8)
        return PILImage.create(noisy_uint8)


class AddRandomMasking(Transform):
    """
    Random rectangular black masking applied only on train split.
    """
    split_idx = 0

    def __init__(self, num_patches: int = 1, patch_size: float = 0.35):
        self.num_patches = num_patches
        self.patch_size = patch_size

    def encodes(self, img: PILImage) -> PILImage:
        img_np = np.array(img).copy()
        h, w, _ = img_np.shape
        ph, pw = max(1, int(h * self.patch_size)), max(1, int(w * self.patch_size))

        for _ in range(self.num_patches):
            top = np.random.randint(0, max(1, h - ph))
            left = np.random.randint(0, max(1, w - pw))
            img_np[top : top + ph, left : left + pw, :] = 0

        return PILImage.create(img_np)
