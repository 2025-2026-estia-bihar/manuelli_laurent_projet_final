from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from fastai.vision.all import (
    ClassificationInterpretation,
    SaveModelCallback,
    EarlyStoppingCallback,
    accuracy,
    F1Score,
    RocAuc,
    vision_learner,
    resnet34,
)

from .augmentations import NoiseConfig, AddRandomNoise
from .callbacks import GradientClipping, NoiseAnnealCallback
from .data import DataConfig, build_dls


def save_metrics(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_training_curves(learn, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = learn.recorder.plot_loss(return_fig=True)
    fig.savefig(out_dir / "training_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def export_confusion_matrix(learn, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    interp = ClassificationInterpretation.from_learner(learn)
    fig = interp.plot_confusion_matrix(return_fig=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train image classifier (fastai) for BIHAR-like dataset.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/image"))
    p.add_argument("--classes", type=int, default=4, choices=[3, 4])

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-pct", type=float, default=0.2)

    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--use-noise", action="store_true")
    p.add_argument("--use-mask", action="store_true")
    p.add_argument("--mask-num-patches", type=int, default=1)
    p.add_argument("--mask-patch-size", type=float, default=0.35)

    p.add_argument("--clip-max-norm", type=float, default=0.5)
    p.add_argument("--noise-anneal", type=float, default=0.8)

    p.add_argument("--early-stop", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        bs=args.bs,
        seed=args.seed,
        valid_pct=args.valid_pct,
        use_noise=args.use_noise,
        use_mask=args.use_mask,
        mask_num_patches=args.mask_num_patches,
        mask_patch_size=args.mask_patch_size,
        noise_cfg=NoiseConfig(),
    )

    dls = build_dls(cfg, classes_mode=args.classes)

    learn = vision_learner(
        dls,
        resnet34,
        metrics=[accuracy, F1Score(), RocAuc()],
        pretrained=args.pretrained,
    )

    cbs = [
        SaveModelCallback(monitor="valid_loss", comp=np.less, fname="best_model"),
        GradientClipping(max_norm=args.clip_max_norm),
        EarlyStoppingCallback(monitor="valid_loss", patience=args.early_stop),
    ]

    # If noise enabled, anneal it away near the end
    if args.use_noise:
        cbs.append(NoiseAnnealCallback(pct_stop=args.noise_anneal, transform_cls=AddRandomNoise))

    learn.fit_one_cycle(args.epochs, lr_max=args.lr, cbs=cbs)

    # export model
    learn.export(out_dir / "model.pkl")

    # artefacts
    export_training_curves(learn, out_dir)
    export_confusion_matrix(learn, out_dir)

    # metrics payload
    payload = {
        "run_id": run_id,
        "classes_mode": args.classes,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "epochs": args.epochs,
        "data_cfg": asdict(cfg),
        "best_model_path": str(out_dir / "best_model.pth"),
        "export_path": str(out_dir / "model.pkl"),
    }
    save_metrics(out_dir, payload)
    print(f"✅ Training finished. Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
