# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/`:
  - `src/main.py` — CLI entry point (run locally).
  - `src/utils.py` — shared helpers.
  - `src/logger.py` — logging utilities.
  - `src/evaluators.py` — evaluation logic.
  - `src/model_clients` — placeholder for model client code.
- Dependencies: `requirements.txt`.
- Create `tests/` for unit tests (see Testing Guidelines).

## Setup, Run, and Dev Commands
- Create venv (PowerShell): `python -m venv venv; .\venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Run CLI: `python src/main.py` (or `python -m src.main`)
- Freeze deps (optional): `pip freeze > requirements.txt`

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indentation; UTF‑8.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `snake_case.py` modules.
- Type hints required for public functions; docstrings for modules and non‑trivial functions.
- Logging via `logger.py`; avoid `print` in library code.
- Formatting/linting (recommended): Black and Ruff — `black src tests` and `ruff check src tests` if configured.

## Testing Guidelines
- Framework: `pytest` (add to dev deps as needed).
- Location: `tests/` with files named `test_*.py`.
- Coverage focus: core logic in `evaluators.py`, `utils.py`, and any client integrations.
- Run tests: `pytest -q` (or `pytest -k <pattern>` for a subset).

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep subject lines to ≤72 chars; body explains rationale and impact.
- PRs must include:
  - Clear description and motivation; reference issues (e.g., `Closes #123`).
  - Summary of changes, any breaking changes, and test coverage notes.
  - Screenshots or CLI output snippets when UX/behavior changes.

## Security & Configuration Tips
- Never commit secrets; read credentials via environment variables (e.g., `os.getenv("API_KEY")`).
- Add local examples as needed (e.g., `.env.example`) and document required variables in `README.md`.

## Agent‑Specific Notes
- Keep file changes minimal and focused; follow structure above.
- Prefer small, composable modules in `src/` and add tests alongside features.
