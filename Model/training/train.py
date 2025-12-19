import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from Model.training.callbacks import CheckpointManager, EarlyStopping
from Model.training.utils import (
    add_repo_to_path,
    build_transforms,
    create_model,
    filter_dataset_by_classes,
    limit_dataset_per_class,
    map_aliases,
    resolve_requested_classes,
)

add_repo_to_path()


OPTIMIZERS = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--train-dir", required=True, help="Path to training images (ImageFolder format).")
    parser.add_argument("--val-dir", required=True, help="Path to validation images (ImageFolder format).")
    parser.add_argument("--model", default="resnet18", choices=["baseline", "resnet18", "resnet34", "resnet50"], help="Model architecture.")
    parser.add_argument("--optimizer", default="adam", choices=list(OPTIMIZERS.keys()))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--augment", choices=["none", "light", "realistic"], default="light", help="Augmentation strategy to apply during training.")
    parser.add_argument("--class-filter", nargs="*", help="Subset of classes to keep (e.g., Chao Ervas Milho).")
    parser.add_argument("--limit-per-class", type=int, help="Limit samples per class for quick experiments.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--monitor", default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--save-path", default="Model/weights/best_model.pt")
    parser.add_argument("--log-dir", default="Monitoring/output")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when available.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    return parser.parse_args()


def build_dataloaders(args):
    train_tfms = build_transforms(args.image_size, augment=args.augment)
    val_tfms = build_transforms(args.image_size, augment="none")

    train_data = datasets.ImageFolder(args.train_dir, transform=train_tfms)
    val_data = datasets.ImageFolder(args.val_dir, transform=val_tfms)

    if args.class_filter:
        keep_train = resolve_requested_classes(train_data.classes, args.class_filter)
        keep_val = resolve_requested_classes(val_data.classes, args.class_filter)
        keep = keep_train or keep_val
        train_data = filter_dataset_by_classes(train_data, keep)
        val_data = filter_dataset_by_classes(val_data, keep)

    if args.limit_per_class:
        train_data = limit_dataset_per_class(train_data, args.limit_per_class)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_data, val_data, train_loader, val_loader


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item() * images.size(0)
        epoch_acc += accuracy(outputs, labels) * images.size(0)
    n = len(loader.dataset)
    return epoch_loss / n, epoch_acc / n


def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * images.size(0)
            epoch_acc += accuracy(outputs, labels) * images.size(0)
    n = len(loader.dataset)
    return epoch_loss / n, epoch_acc / n


def main():
    args = parse_args()
    args.pretrained = False if args.no_pretrained else True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, train_loader, val_loader = build_dataloaders(args)

    num_classes = len(train_data.classes)
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)

    optimizer_cls = OPTIMIZERS[args.optimizer]
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(enabled=args.use_amp)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "logs.txt"

    monitor_mode = "max" if args.monitor.endswith("acc") else "min"
    checkpoint = CheckpointManager(args.save_path, monitor=args.monitor, mode=monitor_mode)
    stopper = EarlyStopping(patience=args.patience, mode=monitor_mode)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)

        monitor_value = metrics[args.monitor]
        improved = checkpoint.save(
            model,
            metric=monitor_value,
            epoch=epoch,
            metadata={
                "class_names": map_aliases(train_data.classes),
                "original_classes": train_data.classes,
                "img_size": args.image_size,
                "model_name": args.model,
                "dropout": args.dropout,
                "pretrained": args.pretrained,
            },
        )

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics, "improved": improved}) + "\n")

        print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
        if stopper.step(monitor_value):
            print("Early stopping triggered.")
            break

    elapsed = time.time() - start
    print(f"Training done in {elapsed/60:.2f} minutes")

    metrics_path = log_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
