# TinyVit overview
- Purpose: Python 3.11 CLI framework to train/eval/predict vision transformer models with Typer, Pydantic v2 configs, and PyTorch.
- Structure: `tinyvit/` package with Typer app in `tinyvit/main.py`; subcommands in `tinyvit/commands/{train,eval,predict}.py`; config models/loaders in `tinyvit/config.py` and `tinyvit/config_loader.py`. Example YAML config at `config.example.yaml`.
- Tooling: Managed with `uv`; lint/format via Ruff; type checks via mypy; pre-commit hooks include nbdev_clean, ruff, ruff-format, mypy. CI runs Ruff lint+format checks.
- Entrypoint: console script `tinyvit` points to `tinyvit.main:app`.
- No tests directory currently present.