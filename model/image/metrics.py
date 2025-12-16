import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def save_metrics(
    y_true: List[int], y_pred: List[int], output_dir: Path, epoch: int | None = None
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    metrics = {"accuracy": float(acc)}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    return metrics
