from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from fastai.vision.all import PILImage, load_learner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/image/lime"))
    p.add_argument("--num-samples", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    learn = load_learner(args.model)
    img = PILImage.create(args.image)
    img_np = np.array(img)

    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "LIME dependencies missing. Install: pip install lime scikit-image matplotlib"
        ) from e

    def predict_fn(images: np.ndarray) -> np.ndarray:
        # images: (N,H,W,3) uint8
        probs = []
        for im in images:
            pil = PILImage.create(im)
            _, _, p = learn.predict(pil)
            probs.append(p.numpy())
        return np.stack(probs, axis=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=predict_fn,
        top_labels=min(4, len(learn.dls.vocab)),
        hide_color=0,
        num_samples=args.num_samples,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(f"LIME explanation – top_label={learn.dls.vocab[top_label]}")
    ax.axis("off")
    out_path = args.output / "lime_explanation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ LIME exported: {out_path}")


if __name__ == "__main__":
    main()
