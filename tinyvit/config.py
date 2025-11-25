"""Configuration models for TinyVit using Pydantic v2."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data configuration."""

    dataset_name: str = Field(description="Name of the dataset")
    batch_size: int = Field(default=32, gt=0, description="Batch size for training")
    num_workers: int = Field(
        default=4, ge=0, description="Number of data loader workers"
    )
    image_size: int = Field(default=224, gt=0, description="Input image size")
    shuffle: bool = Field(
        default=True, description="Shuffle training data (ignored for streaming)"
    )
    dataset_config_name: str | None = Field(
        default=None, description="Optional config name for Hugging Face datasets"
    )
    train_split: str = Field(default="train", description="Train split name")
    val_split: str | None = Field(
        default="validation", description="Validation split name"
    )
    test_split: str | None = Field(default="test", description="Test split name")
    cache_dir: Path | None = Field(
        default=None, description="Cache directory for Hugging Face datasets"
    )
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode to reduce memory/IO for large datasets",
    )


class ModelConfig(BaseModel):
    """Model configuration."""

    model_name: str = Field(description="Name of the timm model architecture")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    num_classes: int = Field(default=10, gt=0, description="Number of output classes")
    drop_rate: float = Field(
        default=0.0, ge=0, le=1, description="Dropout rate for classifier"
    )
    drop_path_rate: float = Field(
        default=0.0, ge=0, le=1, description="Stochastic depth drop path rate"
    )
    checkpoint_path: Path | None = Field(
        default=None, description="Optional checkpoint path to load"
    )
    freeze_backbone: bool = Field(
        default=False,
        description="Freeze backbone parameters, keeping classifier trainable",
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(default=10, gt=0, description="Number of training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    optimizer: Literal["adam", "sgd", "adamw"] = Field(
        default="adam", description="Optimizer type"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cuda", description="Device to use for training"
    )


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    checkpoint_path: Path = Field(description="Path to model checkpoint")
    batch_size: int = Field(default=32, gt=0, description="Batch size for evaluation")
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cuda", description="Device to use for evaluation"
    )


class PredictConfig(BaseModel):
    """Prediction configuration."""

    checkpoint_path: Path = Field(description="Path to model checkpoint")
    input_path: Path = Field(description="Path to input data")
    output_path: Path = Field(description="Path to save predictions")
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cuda", description="Device to use for prediction"
    )


class Config(BaseModel):
    """Main configuration for TinyVit."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig | None = None
    predict: PredictConfig | None = None
