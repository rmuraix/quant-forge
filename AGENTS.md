# Repository Guidelines

## Project Structure & Module Organization

- CLI entry point: `quantforge.cli.app:app` (`quantforge` console script). Commands live in `quantforge/cli/commands/`.
- Configuration schema is in `quantforge/config/schema.py` (Pydantic v2 `ExperimentConfig`). Loading and `--set` overrides are in `quantforge/config/loader.py` and `quantforge/config/overrides.py`. Use `configs/resnet18-cifar10.yaml` as a starting template.
- Key subpackages and their responsibilities:

  | Package | Responsibility |
  |---|---|
  | `quantforge/config/` | Pydantic schema, YAML loader, dot-path overrides, resolved config |
  | `quantforge/data/` | `HFDataModule`, transforms, label mapping, HF dataset loading |
  | `quantforge/models/` | `ModelFactory` (timm), self-contained checkpoint format, metadata |
  | `quantforge/training/` | `Trainer`, train/eval loops, optimizer/scheduler builders, metrics |
  | `quantforge/quantization/` | Strategy pattern: `NoQuantizationStrategy`, `TorchAOQATStrategy`, `TorchAOPTQStrategy` + registry |
  | `quantforge/evaluation/` | `Evaluator`, latency measurement, model size |
  | `quantforge/inference/` | `Predictor` (checkpoint-metadata-driven), preprocessing, output formatting |
  | `quantforge/tracking/` | `WandbTracker` wrapper |
  | `quantforge/artifacts/` | `ArtifactWriter` (all local saves under `outputs/{project}/{run_id}/`) |
  | `quantforge/utils/` | Error hierarchy, device resolution, seeding, Rich console, paths |

- Keep data, model, training, quantization, and evaluation concerns separated. CLI commands should be thin wrappers that delegate to the layer below.

## Build, Test, and Development Commands

- Install deps: `uv sync` (uses `.python-version` for Python version).
- Install with torchao quantization: `uv sync --extra quantize`.
- Run the CLI: `uv run quantforge --help`.
- Common workflow commands:
  ```bash
  uv run quantforge init configs/my-exp.yaml
  uv run quantforge train  --config configs/my-exp.yaml --set training.epochs=5 --set tracking.mode=disabled
  uv run quantforge qat    --config configs/my-exp-qat.yaml
  uv run quantforge eval   --config configs/my-exp.yaml --checkpoint outputs/.../best.pt
  uv run quantforge infer  --checkpoint outputs/.../best.pt --image samples/cat.jpg
  uv run quantforge inspect config  --config configs/my-exp.yaml
  uv run quantforge inspect dataset --config configs/my-exp.yaml
  uv run quantforge inspect model   --config configs/my-exp.yaml
  ```
- Lint: `uv run ruff check quantforge/ tests/`
- Format check: `uv run ruff format --check .`; apply: `uv run ruff format .`
- Type check: `uv run mypy .`
- Tests: `uv run pytest tests/ -v`
- Pre-commit hooks: `uv run pre-commit run --all-files` (includes `nbdev_clean` for notebooks)

## Coding Style & Naming Conventions

- Python 3.11+, Typer CLI, Pydantic v2 configs. Use explicit type hints and clear return types on all public functions.
- Formatting enforced by Ruff; avoid manual style deviations. Fix Ruff lint warnings before merging.
- Naming: `snake_case` for functions/vars/modules; `PascalCase` for classes; CLI command filenames match their verb (e.g. `train.py`, `eval.py`).
- Config YAML keys mirror Pydantic model field names (lowercase with underscores).
- Use `from __future__ import annotations` at the top of every module.
- Use `TYPE_CHECKING` guards for imports that are only needed for type annotations.
- Error messages must be actionable: include the invalid value, what was expected, and a suggested fix.

## Error Handling

- All user-facing errors must be subclasses of `QuantForgeError` (from `quantforge.utils.errors`).
- CLI commands catch `QuantForgeError` and print a user-friendly message via `print_error()`, then exit with code 1.
- Use `--debug` flag (present on every command) to bypass this and show the full traceback.

## Testing Guidelines

- Test files live in `tests/`, named `test_*.py`.
- The test suite currently covers: config loading/validation, override parsing, CLI smoke tests, and artifact writer setup.
- Keep fixtures light (no network I/O, no real datasets, CPU only). Mock external calls (HF datasets, wandb) where possible.
- Run `uv run pytest tests/ -v` plus `uv run ruff check quantforge/ tests/` before opening a PR.

## Quantization Conventions

- `torchao` is an optional dependency (`quantforge[quantize]`). Core code must never import `torchao` at the top level — always use `importlib.util.find_spec("torchao")` or lazy imports inside try/except.
- The quantization strategy registry (`quantforge/quantization/registry.py`) maps `(backend, mode)` tuples to strategy classes. Adding a new backend means: create a strategy class, register it in the registry, and update the schema validator.
- Config validation rules (enforced by `QuantizationConfig.validate_quantization`):
  - `mode` must be `none` when `enabled=false`
  - `backend` must not be `none` when `enabled=true`
  - `mode` in (`qat`, `ptq`) requires `backend=torchao`

## Checkpoint Format

Checkpoints are self-contained dicts saved with `torch.save`. Every checkpoint includes:

```python
{
    "format_version": 1,
    "model_state_dict": ...,
    "model": {"name": ..., "num_classes": ..., "in_chans": ...},
    "preprocess": {"image_size": ..., "mean": ..., "std": ..., "interpolation": ...},
    "label_mapping": {"id_to_label": {...}, "label_to_id": {...}},
    "quantization": {
        "enabled": ...,
        "backend": ...,
        "mode": ...,
        "dtype": ...,
        "recipe": ...,
    },
    "config": {...},  # full ExperimentConfig.model_dump()
    "metrics": {...},
    "environment": {
        "python": ...,
        "torch": ...,
        "torchao": ...,
        "timm": ...,
        "cuda": ...,
    },
}
```

The `infer` command uses only the checkpoint; no config file is required.

## Commit & Pull Request Guidelines

- Commit messages follow conventional commits style: `feat: ...`, `fix: ...`, `chore(deps): ...`, `docs: ...`.
- PRs must include: a short summary, what changed, and how to validate (commands run, configs used).
- CI parity: `uv run ruff check quantforge/ tests/` and `uv run ruff format --check .` must pass locally.
- Update `configs/resnet18-cifar10.yaml`, README, and AGENTS.md when altering CLI flags or config fields.

## Security & Configuration Tips

- Never commit secrets, API keys, dataset paths, or W&B entity names. Use environment variables.
- Validate configs through the CLI before starting long training jobs: `quantforge inspect config --config <file>`.
- Use `--set tracking.mode=disabled` during development to avoid creating wandb runs.
