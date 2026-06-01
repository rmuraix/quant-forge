# QuantForge

A modular CLI toolkit for fine-tuning and quantizing image classification models.

## About

QuantForge provides a clean, configuration-driven workflow for:

1. **Fine-tuning** any [timm](https://github.com/huggingface/pytorch-image-models) model on any [Hugging Face Dataset](https://huggingface.co/docs/datasets)
2. **Quantizing** fine-tuned models via QAT (Quantization-Aware Training) or PTQ (Post-Training Quantization) with [torchao](https://github.com/pytorch/ao)
3. **Evaluating** model accuracy, latency, and size
4. **Running inference** on images directly from a self-contained checkpoint

Experiment tracking is handled by [Weights & Biases](https://wandb.ai/).

## Features

- **Config-driven workflow**: Every experiment is fully described by a single YAML file validated with [Pydantic v2](https://docs.pydantic.dev/)
- **`--set` overrides**: Override any config value at runtime (`--set training.epochs=20`)
- **timm model zoo**: Any `timm`-compatible architecture out of the box
- **HF Datasets**: Load any image-classification dataset from the Hugging Face hub
- **torchao quantization**: QAT and PTQ via `quantforge[quantize]` optional extra
- **Self-contained checkpoints**: Each `.pt` file embeds model arch, preprocessing, label mapping, and environment metadata — inference requires only the checkpoint
- **Structured output**: Every run creates `outputs/{project}/{run_id}/` with checkpoints, metrics, predictions, and manifests
- **Developer tools**: uv, ruff, mypy, pre-commit, nbdev

## Installation

```bash
# Core (no quantization)
pip install -e .

# With torchao quantization support
pip install -e ".[quantize]"

# Or using uv (recommended)
uv sync
uv sync --extra quantize
```

## Quick Start

### 1. Create a config

```bash
quantforge init configs/my-experiment.yaml                  # basic template
quantforge init configs/my-experiment.yaml --template qat   # QAT template
quantforge init configs/my-experiment.yaml --template ptq   # PTQ template
```

### 2. Inspect before training

```bash
quantforge inspect config  --config configs/my-experiment.yaml   # validate YAML
quantforge inspect dataset --config configs/my-experiment.yaml   # check splits/labels
quantforge inspect model   --config configs/my-experiment.yaml   # count parameters
```

### 3. Fine-tune

```bash
quantforge train --config configs/my-experiment.yaml

# Override values at runtime
quantforge train --config configs/my-experiment.yaml \
  --set training.epochs=20 \
  --set training.lr=0.0001 \
  --set tracking.mode=disabled
```

### 4. Quantization

**Quantization-Aware Training (QAT):**

```bash
quantforge qat --config configs/my-experiment-qat.yaml
```

**Post-Training Quantization (PTQ) of an existing checkpoint:**

```bash
quantforge quantize \
  --config configs/my-experiment.yaml \
  --checkpoint outputs/my-project/20240101-120000/checkpoints/best.pt \
  --output outputs/my-project/20240101-120000/quantized/best_int8.pt
```

### 5. Evaluate

```bash
quantforge eval \
  --config configs/my-experiment.yaml \
  --checkpoint outputs/my-project/20240101-120000/checkpoints/best.pt
```

### 6. Inference

```bash
# Single image (no config needed — all metadata is in the checkpoint)
quantforge infer \
  --checkpoint outputs/my-project/20240101-120000/checkpoints/best.pt \
  --image samples/cat.jpg

# Directory of images with JSONL output
quantforge infer \
  --checkpoint outputs/my-project/20240101-120000/checkpoints/best.pt \
  --input-dir samples/ \
  --output predictions.jsonl
```

## Configuration

Every experiment is described by a single YAML file. Use `quantforge init` to scaffold one, then edit as needed.

### Full schema

```yaml
version: 1

project:
  name: resnet18-cifar10   # used to name the output directory
  output_dir: outputs
  seed: 42

model:
  name: resnet18            # any timm model name
  pretrained: true
  num_classes: null         # inferred from dataset when null
  in_chans: 3
  checkpoint_path: null     # set to resume from a specific file

dataset:
  name: cifar10             # HuggingFace dataset name
  subset: null              # dataset config/subset, if any
  split:
    train: train
    validation: test        # split name used for validation
    test: null
  image_column: img         # column name for images
  label_column: label       # column name for labels
  cache_dir: null

preprocess:
  image_size: 224
  mean: null                # defaults to ImageNet mean
  std: null                 # defaults to ImageNet std
  interpolation: bicubic

augment:
  random_resized_crop: true
  horizontal_flip: true
  randaugment: false
  mixup: 0.0
  cutmix: 0.0

training:
  epochs: 10
  batch_size: 64
  lr: 0.0003
  weight_decay: 0.05
  optimizer: adamw          # adamw | adam | sgd
  scheduler: cosine         # cosine | step | none
  warmup_epochs: 1
  amp: true                 # automatic mixed precision
  grad_clip_norm: null
  num_workers: 4
  device: auto              # auto | cuda | cpu
  deterministic: false

quantization:
  enabled: false
  backend: none             # none | torchao
  mode: none                # none | qat | ptq
  dtype: int8
  recipe: null
  calibration_samples: 512

checkpoint:
  resume: null
  save_best: true
  monitor: accuracy

tracking:
  backend: wandb
  project: quantforge
  entity: null
  run_name: null
  group: null
  job_type: null
  tags: []
  notes: null
  mode: online              # online | offline | disabled

logging:
  level: info
  log_interval: 50
  rich: true
```

### Quantization modes

| `backend` | `mode` | Extra required | Description |
|---|---|---|---|
| `none` | `none` | — | No quantization (default) |
| `torchao` | `qat` | `quantforge[quantize]` | Quantization-Aware Training |
| `torchao` | `ptq` | `quantforge[quantize]` | Post-Training Quantization |

### Validation rules

- `quantization.mode` must be `none` when `quantization.enabled=false`
- `quantization.backend` must not be `none` when `quantization.enabled=true`
- `quantization.mode` in (`qat`, `ptq`) requires `quantization.backend=torchao`
- `tracking.backend` must be `wandb` (the only supported backend in v0.1)

## Output structure

Each run creates:

```
outputs/
└── {project.name}/
    └── {run_id}/                    # e.g. 20240101-120000
        ├── config.yaml              # raw experiment config
        ├── config.resolved.yaml     # runtime-resolved config
        ├── label_mapping.json       # id<->label name mapping
        ├── manifest.json            # run summary
        ├── checkpoints/
        │   ├── best.pt              # best val/top1 checkpoint
        │   └── last.pt              # last epoch checkpoint
        ├── quantized/               # quantized models (quantize command)
        ├── metrics/
        │   ├── train.jsonl          # per-epoch training metrics
        │   └── eval.json            # final evaluation metrics
        ├── predictions/             # inference output (infer command)
        └── logs/
```

## Project structure

```
quantforge/
├── cli/                    # Typer commands
│   └── commands/           # init, train, qat, quantize, eval, infer, inspect
├── config/                 # Pydantic schema, YAML loader, --set overrides
├── data/                   # HFDataModule, transforms, label utilities
├── models/                 # timm factory, checkpoint I/O, metadata
├── training/               # Trainer, training/eval loops, optim, schedulers
├── quantization/           # Strategy pattern: NoQuant, TorchAO QAT/PTQ
├── evaluation/             # Evaluator, latency, model-size utilities
├── inference/              # Predictor, preprocessing, output formatting
├── tracking/               # WandbTracker
├── artifacts/              # ArtifactWriter (local file saves)
└── utils/                  # errors, device, seed, logging, rich, paths
```

## Contributing

Your contribution is always welcome. Please read [Contributing Guide](https://github.com/rmuraix/.github/blob/main/.github/CONTRIBUTING.md).
