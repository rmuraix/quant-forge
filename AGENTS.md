# Repository Guidelines

## Project Structure & Module Organization

- CLI entrypoint is `tinyvit/main.py` (`tinyvit` console script). Subcommands live in `tinyvit/commands/{train,eval,predict}.py`.
- Configuration models/loader are in `tinyvit/config.py` and `tinyvit/config_loader.py`. Use `config.example.yaml` as the starting template.
- PyTorch/transformers logic sits behind the CLI layers; keep data/model/training concerns separated when adding new modules.

## Build, Test, and Development Commands

- Install dev deps: `uv sync` (uses `.python-version`).
- Run the CLI: `uv run tinyvit --help`; train with `uv run tinyvit train --config config.example.yaml` (same pattern for `eval` and `predict`).
- Lint: `uv run ruff check .`; format check: `uv run ruff format --check .`; apply formatting with `uv run ruff format .`.
- Type check: `uv run mypy .`.
- Hooks: `uv run pre-commit run --all-files` (includes `nbdev_clean` for notebooks).

## Coding Style & Naming Conventions

- Python 3.11, Typer CLI, Pydantic v2 configs. Prefer explicit type hints and clear return types on CLI functions.
- Formatting is enforced by Ruff formatter; avoid manual style deviations. Follow Ruff lint warnings before merging.
- Naming: snake_case for functions/vars/modules; PascalCase for classes; keep subcommand filenames aligned with CLI verbs (train/eval/predict).
- Config YAML keys mirror Pydantic models; keep names descriptive and lowercase with underscores.

## Testing Guidelines

- No dedicated test suite yet; when adding tests, prefer `pytest` with files named `test_*.py` colocated near code.
- Keep training/eval fixtures light (e.g., tiny datasets, CPU) to avoid slow runs. Mock external I/O where possible.
- Run `uv run pytest` (once added) plus lint/type checks before opening a PR.

## Commit & Pull Request Guidelines

- Commit messages follow a conventional style (e.g., `feat: ...`, `fix: ...`, `chore(deps): ...`), mirroring current history.
- PRs should include: short summary, linked issues/PRs, what changed, and how to validate (commands run, configs used).
- Ensure CI parity locally: `uv run ruff check .` and `uv run ruff format --check .` must pass; include `mypy` results if code touched.
- Update `config.example.yaml` and README/usage snippets when altering CLI flags or config fields.

## Security & Configuration Tips

- Do not commit secrets, API keys, or dataset paths; prefer environment variables and ignore lists.
- Validate configs through the CLI (`tinyvit ... --config <file>`) to catch Pydantic errors early before training jobs run.
