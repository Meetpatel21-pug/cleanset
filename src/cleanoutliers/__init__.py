"""Utilities to detect and remove outliers from tabular and numeric datasets."""

from .core import detect_outliers, remove_outliers

__all__ = ["detect_outliers", "remove_outliers"]
__version__ = "0.1.1"
