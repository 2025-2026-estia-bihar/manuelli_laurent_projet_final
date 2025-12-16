from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NoiseSettings(BaseModel):
    same_across_channels: bool = True
    gaussian_std: float = 0.25
    saltpepper_amount: float = 0.08
    speckle_scale: float = 0.35


class MaskSettings(BaseModel):
    num_patches: int = 1
    patch_size: float = 0.35


class AugmentSettings(BaseModel):
    use_noise: bool = True
    noise: NoiseSettings = Field(default_factory=NoiseSettings)

    use_mask: bool = True
    mask: MaskSettings = Field(default_factory=MaskSettings)

    do_flip: bool = True
    flip_vert: bool = True
    max_rotate: float = 20.0
    max_zoom: float = 1.1
    max_lighting: float = 0.2
    max_warp: float = 0.2
    p_affine: float = 0.75
    p_lighting: float = 0.75


class DataSettings(BaseModel):
    data_dir: Path = Path("data/maize/train")
    classes_mode: Literal[3, 4] = 4
    img_size: int = 224
    batch_size: int = 64
    valid_pct: float = 0.2
    num_workers: int = 4
    per_class_limit: int | None = None
    class_names: list[str] | None = None


class TrainSettings(BaseModel):
    arch: str = "resnet34"
    pretrained: bool = True
    epochs: int = 20
    lr: float = 1e-3
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 0.5
    noise_anneal_pct_stop: float = 0.8


class EvalSettings(BaseModel):
    export_confusion_matrix: bool = True
    export_top_losses: bool = True
    top_losses_k: int = 12


class RunSettings(BaseModel):
    seed: int = 42
    output_dir: Path = Path("artifacts/image")
    run_name: Optional[str] = None


class ImagePipelineSettings(BaseSettings):
    """
    Loads from YAML + environment overrides.
    Priority: env vars > YAML > defaults.
    """
    model_config = SettingsConfigDict(
        env_prefix="BIHAR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    run: RunSettings = Field(default_factory=RunSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    augment: AugmentSettings = Field(default_factory=AugmentSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)


def load_settings(yaml_path: Path) -> ImagePipelineSettings:
    import yaml

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return ImagePipelineSettings.model_validate(raw)
