import numpy as np
import pandas as pd

from cleanoutliers import detect_outliers, remove_outliers


def test_detect_outliers_iqr_array():
    data = np.array([10, 11, 12, 13, 300], dtype=float)
    mask = detect_outliers(data, method="iqr", threshold=1.5)
    assert mask.tolist() == [False, False, False, False, True]


def test_remove_outliers_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3, 100], "b": [2, 3, 4, 5]})
    cleaned = remove_outliers(df, method="iqr", columns=["a"])
    assert len(cleaned) == 3
    assert cleaned["a"].max() == 3


def test_return_mask_shape_matches_rows():
    data = np.array([[1.0, 2.0], [2.0, 2.1], [100.0, 2.2]])
    cleaned, mask = remove_outliers(data, return_mask=True)
    assert mask.shape == (3,)
    assert cleaned.shape[0] == 2
