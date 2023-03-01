import numpy as np
import pandas as pd
import pytest
from DSUtils.data import (
    data_overview,
    generate_target_bins,
    missing_data,
    reduce_mem_usage,
    returnMatches,
    returnNotMatches,
)


def test_data_overview():
    """Test data_overview method"""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [3, "b", 5, "d", 7],
            "D": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    overview = data_overview(df, sort_by_missing=True)

    with pytest.raises(TypeError):
        data_overview("df")

    assert overview.shape == (4, 5)
    assert overview.index.tolist() == ["D", "A", "B", "C"]
    assert overview.columns.tolist() == [
        "Sample",
        "Count uniques",
        "dtype",
        "Count missing",
        "Pct. missing",
    ]
    assert overview.loc["A"].tolist() == ["1", 5, int, 0, 0.0]
    assert overview["Count uniques"].tolist() == [4, 5, 5, 5]
    assert overview["dtype"].tolist() == [float, int, object, object]
    assert overview["Count missing"].tolist() == [1, 0, 0, 0]
    assert overview["Pct. missing"].tolist() == [20.0, 0.0, 0.0, 0.0]


def test_missing_data():
    """Test missing_data method"""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [3, "b", 5, "d", 7],
            "D": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    missing = missing_data(df)

    assert missing.shape == (4, 2)
    assert missing.index.tolist() == ["D", "A", "B", "C"]
    assert missing.columns.tolist() == ["Total", "Percent"]
    assert missing["Total"].tolist() == [1, 0, 0, 0]
    assert missing["Percent"].tolist() == [20.0, 0.0, 0.0, 0.0]


def test_returnNotMatches():
    """Test returnNotMatches method"""

    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    not_matches = returnNotMatches(a, b)

    assert not_matches == [6, 7, 8, 9, 10]


def test_returnMatches():
    """Test returnMatches method"""

    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    matches = returnMatches(a, b)

    assert matches == [1, 2, 3, 4, 5]


def test_reduce_mem_usage():
    """Test reduce_mem_usage method"""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2 ** 10, 4, 6, 8, 10],
            "C": [2 ** 17, 4, 6, 8, 10],
            "D": [2 ** 32, 4, 6, 8, 10],
            "E": [1.0, 2.0, np.nan, 4.0, 5.0],
            "F": [1.0, 2.0 ** 17, 3.0, 4.0, 5.0],
            "G": [1.0, 2.0 ** 250, 3.0, 4.0, 5.0],
            "H": ["a", "b", "c", "d", "e"],
        }
    )

    df = reduce_mem_usage(df, verbose=True)

    with pytest.raises(TypeError):
        reduce_mem_usage("df")

    assert df["A"].dtype == np.int8
    assert df["B"].dtype == np.int16
    assert df["C"].dtype == np.int32
    assert df["D"].dtype == np.int64
    assert df["E"].dtype == np.float16
    assert df["F"].dtype == np.float32
    assert df["G"].dtype == np.float64
    assert df["H"].dtype == object


def test_generate_target_bins():
    """Test generate_target_bins method"""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [3, "b", 5, "d", 7],
            "D": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    # Ensure target is in df
    with pytest.raises(KeyError):
        generate_target_bins([-1, 2, 4, np.inf], df, "E")

    # Ensure bins are a list
    with pytest.raises(TypeError):
        generate_target_bins(-1, df, "A")

    # Ensure target is numeric
    with pytest.raises(TypeError):
        generate_target_bins([1, 2, 3], df, "B")

    # Ensure bins are increasing monotonically
    with pytest.raises(ValueError):
        generate_target_bins([1, 2, 3, 1], df, "A")

    df = generate_target_bins([-1, 2, 4, np.inf], df, "A")
    assert df["A_bins"].tolist() == ["0-2", "0-2", "3-4", "3-4", "5+"]

    df = generate_target_bins([-1, 1, 3], df, "A")
    assert df["A_bins"].tolist() == ["0-1", "2-3", "2-3", np.nan, np.nan]
