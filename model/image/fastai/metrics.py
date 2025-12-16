from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from fastai.vision.all import accuracy
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class MetricsBundle:
    overall: Dict[str, Any]
    per_class: Dict[str, Any]
    confusion_matrix: List[List[int]]
    class_distribution: Dict[str, int]


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics_from_preds(
    probs: torch.Tensor,
    targs: torch.Tensor,
    vocab: List[str],
) -> MetricsBundle:
    probs_np = _as_numpy(probs)
    targs_np = _as_numpy(targs).astype(int)
    pred_np = probs_np.argmax(axis=1)

    # overall
    acc = float((pred_np == targs_np).mean())
    f1_macro = float(f1_score(targs_np, pred_np, average="macro"))
    f1_weighted = float(f1_score(targs_np, pred_np, average="weighted"))

    # roc-auc (multi-class OVR) only if probabilities are well-formed
    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(targs_np, probs_np, multi_class="ovr"))
    except Exception:
        roc_auc = None

    report = classification_report(
        targs_np,
        pred_np,
        target_names=vocab,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(targs_np, pred_np, labels=list(range(len(vocab))))

    # class distribution on targets
    dist = {vocab[i]: int((targs_np == i).sum()) for i in range(len(vocab))}

    overall = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr": roc_auc,
        "support_total": int(len(targs_np)),
    }

    per_class = {cls: report.get(cls, {}) for cls in vocab}

    return MetricsBundle(
        overall=overall,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_distribution=dist,
    )


def save_metrics_json(out_dir: Path, payload: Dict[str, Any]) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
