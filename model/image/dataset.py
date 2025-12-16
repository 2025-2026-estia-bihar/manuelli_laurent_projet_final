import logging
from pathlib import Path
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class SimpleImageFolder(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], image_size: int) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def _find_images(data_dir: Path) -> List[Path]:
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def _normalize(text: str) -> str:
    # Strip accents and lowercase for flexible matching (e.g., "Validação" -> "validacao")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn").casefold()


def _resolve_dir(parent: Path, candidates: Iterable[str]) -> Optional[Path]:
    """Return the first existing directory matching any candidate name (case/accents-insensitive)."""
    if not parent.exists():
        return None
    existing: Dict[str, Path] = {}
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        for key in {p.name.casefold(), _normalize(p.name)}:
            existing.setdefault(key, p)

    for name in candidates:
        for key in {name.casefold(), _normalize(name)}:
            match = existing.get(key)
            if match:
                return match
    return None


def _build_dataset(split_dir: Path, classes: List[str], image_size: int) -> Optional[Dataset]:
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    image_paths: List[Path] = []
    labels: List[int] = []

    for cls in classes:
        class_dir = _resolve_dir(split_dir, [cls])
        if class_dir is None:
            logger.warning("Class directory not found for %s under %s", cls, split_dir)
            continue
        for img_path in _find_images(class_dir):
            image_paths.append(img_path)
            labels.append(label_map[cls])

    if not image_paths:
        return None
    return SimpleImageFolder(image_paths, labels, image_size)


def _load_split_datasets(
    data_dir: Path, classes: List[str], image_size: int
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []

    train_dir = _resolve_dir(data_dir, ["Treino", "Train", "Training"])
    val_dir = _resolve_dir(data_dir, ["Validacao", "Validacion", "Validation", "Validacaoo"])
    test_dir = _resolve_dir(data_dir, ["Teste", "Test"])

    if val_dir is None:
        # Fallback: pick the remaining dir that is not train/test
        for d in all_dirs:
            if train_dir and d == train_dir:
                continue
            if test_dir and d == test_dir:
                continue
            val_dir = d
            logger.warning("Using %s as validation directory fallback", d)
            break

    train_ds = _build_dataset(train_dir, classes, image_size) if train_dir else None
    val_ds = _build_dataset(val_dir, classes, image_size) if val_dir else None
    test_ds = _build_dataset(test_dir, classes, image_size) if test_dir else None
    return train_ds, val_ds, test_ds


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_or_create_dataset(
    data_dir: Path,
    classes: List[str],
    image_size: int,
    batch_size: int = 8,
    num_samples: int = 64,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Prefer explicit train/val/test splits under data_dir; fall back to synthetic/random split.
    """
    train_ds, val_ds, test_ds = _load_split_datasets(data_dir, classes, image_size)

    if train_ds is not None and val_ds is not None:
        train_loader = _make_loader(train_ds, batch_size, shuffle=True)
        val_loader = _make_loader(val_ds, batch_size, shuffle=False)
        test_loader = _make_loader(test_ds, batch_size, shuffle=False) if test_ds else None
        return train_loader, val_loader, test_loader

    logger.warning("No explicit split found; using fallback synthetic/random split from %s", data_dir)
    image_paths = _find_images(data_dir)
    if not image_paths:
        images = torch.randn(num_samples, 3, image_size, image_size)
        labels = torch.randint(low=0, high=len(classes), size=(num_samples,))
        dataset = TensorDataset(images, labels)
    else:
        label_map = {cls: idx for idx, cls in enumerate(classes)}
        labels = []
        for path in image_paths:
            cls_name = path.parent.name
            labels.append(label_map.get(cls_name, 0))
        dataset = SimpleImageFolder(image_paths, labels, image_size)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = _make_loader(train_set, batch_size, shuffle=True)
    val_loader = _make_loader(val_set, batch_size, shuffle=False)
    return train_loader, val_loader, None
