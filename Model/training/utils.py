import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from Model.training.augmentations import get_transforms

CLASS_ALIASES = {
    "chao": "ground",
    "milho": "corn",
    "ervas": "weeds",
    "milho_ervas": "corn_weeds",
    "milho-ervas": "corn_weeds",
}


def add_repo_to_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def normalize_label(label: str) -> str:
    return CLASS_ALIASES.get(label.lower(), label)


def map_aliases(labels: Iterable[str]) -> List[str]:
    return [normalize_label(label) for label in labels]


def build_transforms(image_size: int = 224, augment: str | bool = "none"):
    """
    augment: 'none' | 'light' | 'realistic' (bool True is treated as 'light')
    """
    if isinstance(augment, bool):
        mode = "light" if augment else "none"
    else:
        mode = augment
    return get_transforms(image_size=image_size, mode=mode)


def resolve_requested_classes(all_classes: Sequence[str], requested: Sequence[str]) -> List[str]:
    normalized_all = {normalize_label(cls).lower(): cls for cls in all_classes}
    keep = []
    for req in requested:
        key = normalize_label(req).lower()
        if key in normalized_all:
            keep.append(normalized_all[key])
    return keep


def filter_dataset_by_classes(dataset, keep_classes: Sequence[str]):
    if not keep_classes:
        return dataset
    ordered = [cls for cls in keep_classes if cls in dataset.classes]
    if not ordered:
        return dataset
    class_lookup = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    new_class_to_idx = {cls: i for i, cls in enumerate(ordered)}
    new_samples = []
    for path, target in dataset.samples:
        cls_name = class_lookup[target]
        if cls_name in new_class_to_idx:
            new_samples.append((path, new_class_to_idx[cls_name]))
    dataset.samples = new_samples
    dataset.targets = [t for _, t in new_samples]
    dataset.classes = ordered
    dataset.class_to_idx = new_class_to_idx
    return dataset


def limit_dataset_per_class(dataset, limit: int | None, seed: int = 42):
    if not limit:
        return dataset
    random.seed(seed)
    grouped = {}
    for path, target in dataset.samples:
        grouped.setdefault(target, []).append(path)
    new_samples = []
    for target, paths in grouped.items():
        random.shuffle(paths)
        for path in paths[:limit]:
            new_samples.append((path, target))
    dataset.samples = new_samples
    dataset.targets = [t for _, t in new_samples]
    return dataset


def create_model(
    model_name: str,
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    if model_name == "baseline":
        from Model.architectures.baseline_cnn import create_baseline_cnn

        return create_baseline_cnn(num_classes=num_classes, dropout=dropout)
    from Model.architectures.transfer_resnet import create_resnet

    return create_resnet(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
