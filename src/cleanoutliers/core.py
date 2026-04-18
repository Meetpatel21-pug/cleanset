from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

ArrayLike = Union[np.ndarray, Sequence[float], Iterable[float]]


def _ensure_2d_array(data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Input data must be 1D or 2D numeric data.")
    return arr


def _mask_iqr(arr: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return np.any((arr < lower) | (arr > upper), axis=1)


def _mask_zscore(arr: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, np.nan, std)
    z = np.abs((arr - mean) / std)
    return np.any(z > z_threshold, axis=1)


def _mask_modified_zscore(arr: np.ndarray, z_threshold: float = 3.5) -> np.ndarray:
    median = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - median), axis=0)
    mad = np.where(mad == 0, np.nan, mad)
    modified_z = 0.6745 * np.abs(arr - median) / mad
    return np.any(modified_z > z_threshold, axis=1)


def detect_outliers(
    data: Union[ArrayLike, "pd.DataFrame"],
    method: str = "iqr",
    columns: Optional[Sequence[str]] = None,
    threshold: float = 1.5,
    z_threshold: float = 3.0,
) -> np.ndarray:
    """
    Return a boolean mask where True indicates an outlier row.

    Parameters
    ----------
    data:
        A 1D/2D numeric array-like input or a pandas DataFrame.
    method:
        Outlier detection method: "iqr", "zscore", or "modified_zscore".
    columns:
        DataFrame columns used for outlier detection. Ignored for ndarray input.
    threshold:
        IQR multiplier used when method is "iqr".
    z_threshold:
        Threshold used when method is "zscore" or "modified_zscore".
    """
    method = method.lower().strip()

    if pd is not None and isinstance(data, pd.DataFrame):
        selected = data if columns is None else data.loc[:, list(columns)]
        arr = selected.to_numpy(dtype=float)
    else:
        arr = _ensure_2d_array(data)

    if method == "iqr":
        return _mask_iqr(arr, threshold=threshold)
    if method == "zscore":
        return _mask_zscore(arr, z_threshold=z_threshold)
    if method == "modified_zscore":
        return _mask_modified_zscore(arr, z_threshold=z_threshold)

    raise ValueError('method must be one of: "iqr", "zscore", "modified_zscore"')


def remove_outliers(
    data: Union[ArrayLike, "pd.DataFrame"],
    method: str = "iqr",
    columns: Optional[Sequence[str]] = None,
    threshold: float = 1.5,
    z_threshold: float = 3.0,
    return_mask: bool = False,
):
    """
    Remove outlier rows and return the cleaned dataset.

    If return_mask is True, returns a tuple: (cleaned_data, outlier_mask).
    """
    mask = detect_outliers(
        data=data,
        method=method,
        columns=columns,
        threshold=threshold,
        z_threshold=z_threshold,
    )

    if pd is not None and isinstance(data, pd.DataFrame):
        cleaned = data.loc[~mask].copy()
    else:
        arr = _ensure_2d_array(data)
        cleaned = arr[~mask]
        if np.asarray(data).ndim == 1:
            cleaned = cleaned.reshape(-1)

    return (cleaned, mask) if return_mask else cleaned
