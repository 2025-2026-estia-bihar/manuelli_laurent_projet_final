from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from fastai.vision.all import (
    ClassificationInterpretation,
    EarlyStoppingCallback,
    SaveModelCallback,
    accuracy,
    vision_learner,
    resnet34,
    resnet18,
    resnet50,
)

from .augmentations import NoiseConfig, AddRandomNoise
from .callbacks import GradientClipping, NoiseAnnealCallback
from .data import DataConfig, build_dls
from .metrics import compute_metrics_from_preds, save_metrics_json
from .settings import load_settings


ARCH_MAP = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def export_figures(learn, out_dir: Path, top_losses_k: int = 12):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # training curve
    ax = learn.recorder.plot_loss()
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
    fig.savefig(out_dir / "training_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    fig = plt.gcf()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # top losses (optional)
    try:
        interp.plot_top_losses(top_losses_k, nrows=max(1, top_losses_k // 4))
        fig = plt.gcf()
        fig.savefig(out_dir / "top_losses.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def main():
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_settings(cfg_path)

    run_id = cfg.run.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.run.output_dir / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # build DataConfig from YAML
    data_cfg = DataConfig(
        data_dir=cfg.data.data_dir,
        img_size=cfg.data.img_size,
        bs=cfg.data.batch_size,
        seed=cfg.run.seed,
        valid_pct=cfg.data.valid_pct,
        num_workers=cfg.data.num_workers,
        per_class_limit=cfg.data.per_class_limit,
        class_names=cfg.data.class_names,
        use_noise=cfg.augment.use_noise,
        noise_cfg=NoiseConfig(
            same_across_channels=cfg.augment.noise.same_across_channels,
            gaussian_std=cfg.augment.noise.gaussian_std,
            saltpepper_amount=cfg.augment.noise.saltpepper_amount,
            speckle_scale=cfg.augment.noise.speckle_scale,
        ),
        use_mask=cfg.augment.use_mask,
        mask_num_patches=cfg.augment.mask.num_patches,
        mask_patch_size=cfg.augment.mask.patch_size,
        do_flip=cfg.augment.do_flip,
        flip_vert=cfg.augment.flip_vert,
        max_rotate=cfg.augment.max_rotate,
        max_zoom=cfg.augment.max_zoom,
        max_lighting=cfg.augment.max_lighting,
        max_warp=cfg.augment.max_warp,
        p_affine=cfg.augment.p_affine,
        p_lighting=cfg.augment.p_lighting,
    )

    dls = build_dls(data_cfg, classes_mode=cfg.data.classes_mode)

    arch = ARCH_MAP.get(cfg.train.arch)
    if not arch:
        raise ValueError(f"Unsupported arch '{cfg.train.arch}'. Supported: {list(ARCH_MAP)}")

    learn = vision_learner(
        dls,
        arch,
        metrics=[accuracy],
        pretrained=cfg.train.pretrained,
    )

    callbacks = [
        SaveModelCallback(monitor="valid_loss", comp=np.less, fname="best_model"),
        GradientClipping(max_norm=cfg.train.grad_clip_max_norm),
        EarlyStoppingCallback(monitor="valid_loss", patience=cfg.train.early_stop_patience),
    ]
    if cfg.augment.use_noise:
        callbacks.append(
            NoiseAnnealCallback(
                pct_stop=cfg.train.noise_anneal_pct_stop,
                transform_cls=AddRandomNoise,
            )
        )

    learn.fit_one_cycle(cfg.train.epochs, lr_max=cfg.train.lr, cbs=callbacks)

    # export model
    learn.export(out_dir / "model.pkl")

    # compute enriched metrics on valid set
    probs, targs = learn.get_preds(dl=dls.valid)
    bundle = compute_metrics_from_preds(probs=probs, targs=targs, vocab=list(dls.vocab))

    # figures
    if cfg.eval.export_confusion_matrix or cfg.eval.export_top_losses:
        export_figures(learn, out_dir, top_losses_k=cfg.eval.top_losses_k)

    payload = {
        "run_id": run_id,
        "settings": cfg.model_dump(),
        "valid_metrics": bundle.overall,
        "per_class": bundle.per_class,
        "class_distribution_valid": bundle.class_distribution,
        "confusion_matrix_valid": bundle.confusion_matrix,
        "artifacts": {
            "model_export": str(out_dir / "model.pkl"),
            "training_loss": str(out_dir / "training_loss.png"),
            "confusion_matrix": str(out_dir / "confusion_matrix.png"),
            "top_losses": str(out_dir / "top_losses.png"),
        },
    }
    save_metrics_json(out_dir, payload)
    print(f"✅ Run complete: {out_dir}")


if __name__ == "__main__":
    main()
