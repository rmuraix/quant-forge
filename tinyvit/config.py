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


class ModelConfig(BaseModel):
    """Model configuration."""

    model_name: str = Field(description="Name of the model architecture")
    pretrained: bool = Field(default=False, description="Use pretrained weights")
    num_classes: int = Field(default=10, gt=0, description="Number of output classes")


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
