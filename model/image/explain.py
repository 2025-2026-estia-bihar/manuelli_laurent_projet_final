from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image


def explain_with_lime(
    image_path: Path,
    predict_fn: Callable[[Path], List[dict[str, Any]]],
    top_labels: int = 2,
    num_samples: int = 1000,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a LIME explanation image for a single prediction.

    predict_fn must take a Path and return a list of {"label": str, "score": float}.
    """
    try:
        from lime import lime_image  # type: ignore
        from skimage.segmentation import mark_boundaries  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return "LIME not installed; run `pip install lime scikit-image matplotlib` to enable explanations"

    image = np.array(Image.open(image_path).convert("RGB"))

    def classifier_fn(images: np.ndarray):
        outputs = []
        for arr in images:
            tmp = Image.fromarray(arr.astype(np.uint8))
            preds = predict_fn(tmp)  # predict_fn must accept a PIL image or Path-like
            outputs.append([p["score"] for p in preds])
        return np.array(outputs)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        classifier_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=False, num_features=10, hide_rest=False
    )
    output_path = output_path or image_path.with_name(f"{image_path.stem}_lime.png")
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return f"LIME explanation saved to {output_path}"
