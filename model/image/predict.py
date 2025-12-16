import argparse
import logging
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from .models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _latest_model(registry_dir: Path) -> Path:
    if not registry_dir.exists():
        raise FileNotFoundError(f"Registry not found at {registry_dir}")
    runs = sorted(registry_dir.glob("image_*"), reverse=True)
    if not runs:
        raise FileNotFoundError("No models found in registry")
    return runs[0] / "model.pt"


def load_model(model_path: Path, num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = create_model(num_classes=num_classes, dropout=dropout, use_pretrained=False)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


ImageInput = Union[Path, Image.Image]


def _prepare_tensor(image_input: ImageInput) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor


def predict(image: ImageInput, model_path: Path, classes: List[str], top_k: int = 3):
    tensor = _prepare_tensor(image)

    model = load_model(model_path, num_classes=len(classes))
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(classes)))
    return [
        {"label": classes[idx], "score": float(prob)}
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to model weights")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Chao", "Milho", "Ervas", "Milho_ervas"],
        help="Class labels",
    )
    parser.add_argument("--registry", type=Path, default=Path("model/registry"), help="Registry dir")
    parser.add_argument("--top-k", type=int, default=3, help="Top K predictions")
    args = parser.parse_args()

    model_path = args.model_path or _latest_model(args.registry)
    results = predict(args.image, model_path, classes=args.classes, top_k=args.top_k)
    for item in results:
        print(f"{item['label']}: {item['score']:.3f}")


if __name__ == "__main__":
    main()
