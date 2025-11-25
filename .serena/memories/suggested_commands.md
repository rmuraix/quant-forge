# Suggested commands
- Install deps (dev): `uv sync` (uses Python version from `.python-version`).
- Run CLI in editable mode: `uv run tinyvit --help` (after `pip install -e .` or `uv sync`).
- Train/eval/predict: `uv run tinyvit train --config config.example.yaml` (similarly `eval`, `predict`).
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .` (CI enforces `--check`)
- Type check: `uv run mypy .`
- Pre-commit locally: `uv run pre-commit run --all-files`
- Clean notebooks (if any): `uv run nbdev_clean`