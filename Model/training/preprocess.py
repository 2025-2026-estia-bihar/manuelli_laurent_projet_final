import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def process_split(split_dir: Path, output_dir: Path, size: int, limit_per_class: int | None) -> None:
    if not split_dir.exists():
        return
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        target_dir = output_dir / f"{split_dir.name}_{size}" / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in class_dir.iterdir() if p.is_file()])
        if limit_per_class:
            images = images[:limit_per_class]
        for img_path in tqdm(images, desc=f"{split_dir.name}-{class_dir.name}"):
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
                img.save(target_dir / img_path.name)


def main():
    parser = argparse.ArgumentParser(description="Resize and copy images into processed folders.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root with train/val/test folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/processed"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()

    for split in ["train", "val", "test", "Treino", "Teste"]:
        process_split(args.input_dir / split, args.output_dir, args.size, args.limit_per_class)


if __name__ == "__main__":
    main()
