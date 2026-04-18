# cleanoutliers

A lightweight Python library to detect and remove outlier rows from arrays and DataFrames.

## Features

- Remove outliers from NumPy arrays and pandas DataFrames.
- Multiple methods: IQR, Z-score, Modified Z-score.
- Optionally return the outlier mask for inspection.

## Installation

```bash
pip install cleanoutliers
```

For pandas support:

```bash
pip install "cleanoutliers[pandas]"
```

## Quick Start

```python
import numpy as np
from cleanoutliers import remove_outliers

data = np.array([10, 11, 12, 13, 300])
cleaned = remove_outliers(data, method="iqr", threshold=1.5)
print(cleaned)  # [10. 11. 12. 13.]
```

### With pandas DataFrame

```python
import pandas as pd
from cleanoutliers import remove_outliers

df = pd.DataFrame({"age": [20, 21, 22, 99], "score": [88, 90, 91, 89]})
clean_df = remove_outliers(df, method="iqr", columns=["age"])
```

## API

### `detect_outliers(data, method="iqr", columns=None, threshold=1.5, z_threshold=3.0)`
Returns a boolean mask where `True` means the row is an outlier.

### `remove_outliers(data, method="iqr", columns=None, threshold=1.5, z_threshold=3.0, return_mask=False)`
Returns cleaned data with outlier rows removed.
If `return_mask=True`, returns `(cleaned_data, outlier_mask)`.

## Publish to PyPI

1. Update metadata in `pyproject.toml` (name, author, email).
2. Install publishing tools:
   ```bash
   pip install -U build twine
   ```
3. Build package:
   ```bash
   python -m build
   ```
4. Check distributions:
   ```bash
   twine check dist/*
   ```
5. Upload to TestPyPI (recommended first):
   ```bash
   twine upload --repository testpypi dist/*
   ```
6. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

## Publish with GitHub Trusted Publishing (No API Token)

This project includes GitHub Actions workflows for Trusted Publishing:

- `.github/workflows/publish-testpypi.yml`
- `.github/workflows/publish-pypi.yml`

### One-time setup on PyPI/TestPyPI

1. Create accounts on PyPI and TestPyPI.
2. In each project settings page, add a Trusted Publisher.
3. Provider: GitHub.
4. Fill in:
   - GitHub owner/user
   - Repository name
   - Workflow filename (`publish-testpypi.yml` for TestPyPI, `publish-pypi.yml` for PyPI)
   - Environment: leave empty unless you use one

### Release flow

1. Commit and push your changes to GitHub.
2. For a TestPyPI pre-release, create a release-candidate tag:
   ```bash
   git tag v0.1.0-rc1
   git push origin v0.1.0-rc1
   ```
3. For a PyPI release, create a stable tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

The corresponding GitHub workflow will build and publish automatically using OIDC.

## License

MIT
