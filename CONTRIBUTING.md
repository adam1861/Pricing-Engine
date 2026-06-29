# Contributing

## Setup

1. Create and activate a virtual environment.
2. Install the project with development dependencies:

```bash
pip install .[dev]
```

3. Download the Kaggle dataset referenced in [README.md](README.md) and place the required files in `Data/raw/`.

## Before Opening a Pull Request

Run the checks locally:

```bash
ruff check .
pytest
```

## Pull Request Expectations

- Keep changes focused and easy to review.
- Update documentation when behavior or setup changes.
- Add or update tests when pricing logic changes.
- Do not commit raw or processed dataset files.
