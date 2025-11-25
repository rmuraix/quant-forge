# TinyVit

A modular Vision Transformer training framework with a clean CLI interface.

## About

TinyVit provides a modular, easy-to-use command-line interface for training, evaluating, and making predictions with vision transformer models. The framework uses Typer for CLI management and Pydantic v2 for robust configuration validation.

## Features

- **Modular CLI**: Train, evaluate, and predict subcommands using [Typer](https://typer.tiangolo.com/)
- **Type-safe Configuration**: YAML-based configuration with [Pydantic v2](https://docs.pydantic.dev/) validation
- **Well-structured Code**: Decoupled modules following best practices
- **Developer Tools**: 
  - Run container as a non-root user
  - Package management with [uv](https://docs.astral.sh/uv/)
  - Linting, code formatting and type checking with [ruff](https://docs.astral.sh/ruff/) and [mypy](https://www.mypy-lang.org/)
  - Clean notebooks to avoid merge conflicts with [nbdev](https://nbdev.fast.ai/) and [pre-commit](https://pre-commit.com/)

## Installation

```bash
pip install -e .
```

## Usage

TinyVit provides three main commands:

### Train

Train a model using a configuration file:

```bash
tinyvit train --config config.yaml
```

### Evaluate

Evaluate a trained model:

```bash
tinyvit eval --config config.yaml
```

### Predict

Make predictions with a trained model:

```bash
tinyvit predict --config config.yaml
```

## Configuration

The configuration is defined in YAML format and validated using Pydantic v2.

### Basic Structure

```yaml
data:
  dataset_name: "cifar10"
  batch_size: 32
  num_workers: 4

model:
  model_name: "tiny_vit"
  pretrained: false
  num_classes: 10

training:
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"  # adam, sgd, or adamw
  device: "cuda"     # cuda, cpu, or mps

# Optional sections
eval:
  checkpoint_path: "./checkpoints/best_model.pt"
  batch_size: 64
  device: "cuda"

predict:
  checkpoint_path: "./checkpoints/best_model.pt"
  input_path: "./data/test_images"
  output_path: "./predictions/results.json"
  device: "cuda"
```

### Configuration Validation

All configuration fields are validated using Pydantic v2:
- Required fields are enforced
- Type checking ensures correct data types
- Constraints validate value ranges (e.g., batch_size > 0)
- Clear error messages for invalid configurations

## Contributing

Your contribution is always welcome. Please read [Contributing Guide](https://github.com/rmuraix/.github/blob/main/.github/CONTRIBUTING.md).
