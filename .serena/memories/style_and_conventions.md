# Style and conventions
- Language: Python 3.11; Typer CLI + Pydantic v2 configs; prefers type hints and dataclass-like models in `config.py`.
- Formatting: Ruff formatter (`uv run ruff format .`); lint via `ruff check .`; follow its rules.
- Types: mypy configured via pre-commit; aim for typed functions/CLI params.
- Naming: snake_case for functions/vars/files; PascalCase for classes; CLI commands live in `tinyvit/commands/*.py` mirroring subcommand names.
- Configs: YAML files validated via Pydantic models; see `config.example.yaml` for required fields.
- Commits: History shows `fix(deps): ...` style; keep concise conventional-prefix messages (e.g., `feat:`, `fix:`, `chore:`) referencing PR/issue when relevant.