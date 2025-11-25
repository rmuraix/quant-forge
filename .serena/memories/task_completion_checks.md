# Task completion checks
- Code installs with `uv sync` and still runs primary CLI commands (train/eval/predict) without errors.
- Lint clean: `uv run ruff check .`
- Format clean: `uv run ruff format --check .`
- Type check (if touching Python): `uv run mypy .`
- Optional: run `uv run pre-commit run --all-files` for all hooks (includes nbdev_clean if notebooks exist).
- Update config examples/docs if CLI flags or config schema change.