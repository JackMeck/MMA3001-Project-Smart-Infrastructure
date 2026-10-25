# MMA3001 Smart Infrastructure Project

This repository contains the Monash Smart Infrastructure occupancy-prediction work in a cleaner project layout.

## Repository layout

```text
MMA3001-Project-Smart-Infrastructure/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── smart_infrastructure/
│       ├── __init__.py
│       └── project.py
├── notebooks/
│   └── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── results/
│   └── README.md
├── tests/
│   └── test_imports.py
├── archive/
│   ├── README
│   └── project.py
├── Final_Project.ipynb
├── MMA3001_Project.ipynb
├── Project.ipynb
└── LICENSE
```

## Why this structure

- `src/smart_infrastructure/` is the canonical Python package for reusable code.
- `notebooks/` is reserved for notebook-based analysis and reporting.
- `data/` keeps input datasets and processed outputs separated from source code.
- `results/` stores generated figures, metrics and final reports.
- `archive/` preserves the legacy notebook-converted script for reference.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Then import the package using:

```python
import smart_infrastructure
from smart_infrastructure import project
```

## Notes

The original notebook files remain in the repository root so the project history and earlier analysis remain visible and accessible. The reorganised package layout is intended to make future development, testing, and publication cleaner while keeping the original work intact.
