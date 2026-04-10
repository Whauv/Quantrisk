# AGENTS

## Project Purpose

Quantrisk is a Python analytics package and Streamlit dashboard for regime-aware portfolio analysis.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Common Commands

Run the dashboard:

```powershell
python -m streamlit run src\quantrisk\dashboard\app.py
```

Run tests:

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

Compile-check key modules:

```powershell
python -m py_compile src\quantrisk\*.py src\quantrisk\dashboard\*.py
```

## Folder Map

- `src/quantrisk/`: core package source
- `src/quantrisk/dashboard/`: Streamlit UI, chart helpers, styling, and assets
- `tests/`: test suite and mirrored placeholders for package modules
- `data/`: generated local artifacts and cached pipeline outputs
- `notebooks/`: exploratory notebooks and research materials

## Code Style

- Prefer explicit types and docstrings on public functions and classes.
- Keep analytics logic in package modules, not in the dashboard entrypoint.
- Avoid hardcoding credentials; use environment variables and `.env.example`.
- Treat `data/` as generated runtime output unless explicitly designated as sample data.

## Refactor Guardrails

- Preserve business logic behavior unless the task explicitly allows behavioral changes.
- Use `git mv` for file moves so history remains intact.
- Fix imports immediately after structural changes.
- Keep tests green after every batch of changes.
