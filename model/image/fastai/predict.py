from __future__ import annotations

import argparse
from pathlib import Path

from fastai.vision.all import PILImage, load_learner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to exported fastai model.pkl")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--topk", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    learn = load_learner(args.model)
    img = PILImage.create(args.image)
    pred, pred_idx, probs = learn.predict(img)

    topk = min(args.topk, len(probs))
    pairs = sorted(
        [(learn.dls.vocab[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True,
    )[:topk]

    print(f"Pred: {pred} (idx={pred_idx})")
    for cls, p in pairs:
        print(f"- {cls}: {p:.4f}")


if __name__ == "__main__":
    main()
