import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from Model.training.utils import (
    add_repo_to_path,
    build_transforms,
    create_model,
    filter_dataset_by_classes,
    map_aliases,
    resolve_requested_classes,
)

add_repo_to_path()


def plot_confusion(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def maybe_plot_training_curves(metrics_json: Path, save_path: Path | None):
    if not save_path or not metrics_json.exists():
        return
    history = json.loads(metrics_json.read_text())
    epochs = [m["epoch"] for m in history]
    train_loss = [m["train_loss"] for m in history]
    val_loss = [m["val_loss"] for m in history]
    train_acc = [m["train_acc"] for m in history]
    val_acc = [m["val_acc"] for m in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder directory to evaluate (e.g., test split).")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--model", default="resnet18", help="Model name used during training.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--class-filter", nargs="*", help="Subset of classes to keep.")
    parser.add_argument("--confusion-path", type=Path, default=Path("Visualisation/confusion_matrix.png"))
    parser.add_argument("--report-path", type=Path, default=Path("Monitoring/output/metrics.json"))
    parser.add_argument("--metrics-json", type=Path, help="Training metrics.json for plotting curves.")
    parser.add_argument("--training-curves", type=Path, help="Where to save training curve plot.")
    return parser.parse_args()


def load_model(args, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=args.model,
        num_classes=len(class_names),
        dropout=args.dropout,
        pretrained=False,
        freeze_backbone=False,
    )
    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=False)
    metadata = checkpoint.get("metadata", {})
    model_name = metadata.get("model_name")
    if model_name:
        args.model = model_name
    img_size = metadata.get("img_size")
    if img_size:
        args.image_size = img_size
    model.to(device)
    model.eval()
    return model, device


def main():
    args = parse_args()
    tfms = build_transforms(args.image_size, augment="none")
    dataset = datasets.ImageFolder(args.data_dir, transform=tfms)
    if args.class_filter:
        keep = resolve_requested_classes(dataset.classes, args.class_filter)
        dataset = filter_dataset_by_classes(dataset, keep)
    class_names = map_aliases(dataset.classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, device = load_model(args, class_names)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_confusion(y_true, y_pred, class_names, args.confusion_path)
    maybe_plot_training_curves(args.metrics_json, args.training_curves)
    print(f"Evaluation done. Report saved to {args.report_path}")


if __name__ == "__main__":
    main()
