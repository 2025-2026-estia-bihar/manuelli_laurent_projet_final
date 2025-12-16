from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    RandomSplitter,
    Resize,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    Normalize,
)

from .augmentations import AddRandomMasking, AddRandomNoise, NoiseConfig


@dataclass
class DataConfig:
    data_dir: Path
    img_size: int = 224
    bs: int = 64
    seed: int = 42
    valid_pct: float = 0.2
    num_workers: int = 4
    per_class_limit: Optional[int] = None  # cap images per class to shrink train time
    class_names: Optional[List[str]] = None  # explicit vocab if folder names differ

    # augmentation toggles
    use_noise: bool = True
    noise_cfg: NoiseConfig = NoiseConfig()
    use_mask: bool = True
    mask_num_patches: int = 1
    mask_patch_size: float = 0.35

    # fastai default aug
    do_flip: bool = True
    flip_vert: bool = True
    max_rotate: float = 20.0
    max_zoom: float = 1.1
    max_lighting: float = 0.2
    max_warp: float = 0.2
    p_affine: float = 0.75
    p_lighting: float = 0.75


def infer_vocab_for_maize(classes_mode: int) -> List[str]:
    """
    BIHAR: start with 3 classes then 4 classes.
    Expect folders with names:
    - ground
    - corn
    - weeds
    - corn_weeds (or corn/weeds depending on your naming)
    """
    if classes_mode == 3:
        return ["ground", "corn", "weeds"]
    if classes_mode == 4:
        # normalize folder naming: prefer "corn_weeds"
        return ["ground", "corn", "weeds", "corn_weeds"]
    raise ValueError("classes_mode must be 3 or 4")


def parent_label(p: Path) -> str:
    return p.parent.name


def limit_per_class(
    items: List[Path],
    label_fn: Callable[[Path], str],
    vocab_set: set[str],
    per_class_limit: int,
) -> List[Path]:
    buckets: Dict[str, List[Path]] = {}
    for p in sorted(items):
        cls = label_fn(p)
        if cls not in vocab_set:
            continue
        bucket = buckets.setdefault(cls, [])
        if len(bucket) < per_class_limit:
            bucket.append(p)
    return [p for paths in buckets.values() for p in paths]


def build_dls(
    cfg: DataConfig,
    classes_mode: int = 4,
    label_func: Optional[Callable[[Path], str]] = None,
):
    """
    Folder-based dataset expected:
      data_dir/
        class_a/...
        class_b/...
    If label_func provided, it overrides parent-folder labeling.
    """
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    vocab = cfg.class_names or infer_vocab_for_maize(classes_mode)
    vocab_set = set(vocab)

    label_fn = label_func or parent_label

    # collect files; optionally limit per class for faster runs
    items = get_image_files(data_dir)
    if cfg.per_class_limit:
        items = limit_per_class(items, label_fn, vocab_set, cfg.per_class_limit)

    def get_items_fn(_: Path) -> List[Path]:
        return items

    item_tfms: Sequence = [Resize(cfg.img_size)]
    # train-only custom augs go here (PIL stage)
    if cfg.use_noise:
        item_tfms = [AddRandomNoise(cfg.noise_cfg), *item_tfms]
    if cfg.use_mask:
        item_tfms = [AddRandomMasking(cfg.mask_num_patches, cfg.mask_patch_size), *item_tfms]

    batch_tfms = [
        *aug_transforms(
            do_flip=cfg.do_flip,
            flip_vert=cfg.flip_vert,
            max_rotate=cfg.max_rotate,
            max_zoom=cfg.max_zoom,
            max_lighting=cfg.max_lighting,
            max_warp=cfg.max_warp,
            p_affine=cfg.p_affine,
            p_lighting=cfg.p_lighting,
        ),
        Normalize.from_stats(*imagenet_stats),
    ]

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
        get_items=get_items_fn,
        get_y=label_fn,
        splitter=RandomSplitter(valid_pct=cfg.valid_pct, seed=cfg.seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = dblock.dataloaders(data_dir, bs=cfg.bs, num_workers=cfg.num_workers)
    return dls
