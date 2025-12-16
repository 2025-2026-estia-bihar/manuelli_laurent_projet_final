import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Adam, SGD

from .dataset import load_or_create_dataset
from .metrics import save_metrics
from .models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _select_optimizer(name: str, parameters, lr: float):
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr)
    if name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def train(
    data_dir: Path,
    classes: List[str],
    epochs: int,
    image_size: int,
    batch_size: int,
    optimizer_name: str,
    dropout: float,
    use_pretrained: bool,
    output_dir: Path,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    train_loader, val_loader, test_loader = load_or_create_dataset(
        data_dir, classes, image_size, batch_size=batch_size
    )

    model = create_model(num_classes=len(classes), dropout=dropout, use_pretrained=use_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _select_optimizer(optimizer_name, model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info("Epoch %s/%s completed", epoch + 1, epochs)

    val_labels, val_preds = _evaluate(model, val_loader, device)
    test_labels: Optional[List[int]] = None
    test_preds: Optional[List[int]] = None
    if test_loader is not None:
        test_labels, test_preds = _evaluate(model, test_loader, device)

    registry_dir = output_dir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    git_sha = os.getenv("GIT_COMMIT_SHA", "nogit")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = registry_dir / f"image_{timestamp}_{git_sha}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    save_metrics(val_labels, val_preds, run_dir / "val")
    if test_labels and test_preds:
        save_metrics(test_labels, test_preds, run_dir / "test")

    summary = {
        "classes": classes,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "dropout": dropout,
        "use_pretrained": use_pretrained,
        "model_path": model_path.as_posix(),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels) if test_labels else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete; artifacts at %s", run_dir)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset directory")
    parser.add_argument("--classes", nargs="+", default=["cat", "dog"], help="List of classes")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--output-dir", type=Path, default=Path("model"), help="Output directory")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        classes=args.classes,
        epochs=args.epochs,
        image_size=args.img_size,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        dropout=args.dropout,
        use_pretrained=args.use_pretrained,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
