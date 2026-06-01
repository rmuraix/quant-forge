"""Pydantic v2 config schema for QuantForge experiments."""

from __future__ import annotations


from pydantic import BaseModel, field_validator, model_validator


class SplitConfig(BaseModel):
    train: str = "train"
    validation: str = "validation"
    test: str | None = None


class ProjectConfig(BaseModel):
    name: str = "experiment"
    output_dir: str = "outputs"
    seed: int = 42


class ModelConfig(BaseModel):
    name: str = "resnet18"
    pretrained: bool = True
    num_classes: int | None = None
    in_chans: int = 3
    checkpoint_path: str | None = None


class DatasetConfig(BaseModel):
    name: str = "cifar10"
    subset: str | None = None
    split: SplitConfig = SplitConfig()
    image_column: str = "img"
    label_column: str = "label"
    cache_dir: str | None = None


class PreprocessConfig(BaseModel):
    image_size: int = 224
    mean: list[float] | None = None
    std: list[float] | None = None
    interpolation: str = "bicubic"


class AugmentConfig(BaseModel):
    random_resized_crop: bool = True
    horizontal_flip: bool = True
    randaugment: bool = False
    mixup: float = 0.0
    cutmix: float = 0.0


class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.0003
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    amp: bool = True
    grad_clip_norm: float | None = None
    num_workers: int = 4
    device: str = "auto"
    deterministic: bool = False


class QuantizationConfig(BaseModel):
    enabled: bool = False
    backend: str = "none"
    mode: str = "none"
    dtype: str = "int8"
    recipe: str | None = None
    calibration_samples: int = 512

    @model_validator(mode="after")
    def validate_quantization(self) -> "QuantizationConfig":
        if not self.enabled and self.mode != "none":
            raise ValueError(
                "quantization.mode must be 'none' when quantization.enabled=false"
            )
        if self.enabled and self.backend == "none":
            raise ValueError(
                "quantization.backend must not be 'none' when quantization.enabled=true"
            )
        if self.mode in ("qat", "ptq") and self.backend != "torchao":
            raise ValueError(
                f"quantization.mode='{self.mode}' requires quantization.backend='torchao'"
            )
        return self


class CheckpointConfig(BaseModel):
    resume: str | None = None
    save_best: bool = True
    monitor: str = "accuracy"


class TrackingConfig(BaseModel):
    backend: str = "wandb"
    project: str = "quantforge"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    job_type: str | None = None
    tags: list[str] = []
    notes: str | None = None
    mode: str = "online"
    log_config: bool = True
    log_metrics: bool = True
    log_system: bool = True
    log_predictions: bool = False
    max_prediction_samples: int = 32
    watch_model: bool = False
    upload_artifacts: bool = False
    upload_checkpoints: bool = False
    upload_quantized_models: bool = False

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v != "wandb":
            raise ValueError(
                f"tracking.backend='{v}' is not supported. Only 'wandb' is supported in v0.1."
            )
        return v


class LoggingConfig(BaseModel):
    level: str = "info"
    log_interval: int = 50
    rich: bool = True


class ExperimentConfig(BaseModel):
    version: int = 1
    project: ProjectConfig = ProjectConfig()
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    augment: AugmentConfig = AugmentConfig()
    training: TrainingConfig = TrainingConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    tracking: TrackingConfig = TrackingConfig()
    logging: LoggingConfig = LoggingConfig()
