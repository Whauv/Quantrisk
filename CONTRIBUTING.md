# Contributing

## Development Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Before Opening a Pull Request

1. Run the test suite.
2. Verify the Streamlit entrypoint still starts.
3. Keep generated artifacts out of commits unless they are intentionally tracked sample outputs.

## Testing

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

## Style Expectations

- Add docstrings and type hints to public code.
- Keep dashboard presentation code separate from analytics logic.
- Prefer small, composable modules over large monolithic files.
