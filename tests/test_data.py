import numpy as np
import pandas as pd
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

    overview = data_overview(df)

    assert overview.shape == (4, 5)
    assert overview.index.tolist() == ["A", "B", "C", "D"]
    assert overview.columns.tolist() == [
        "Sample",
        "Count uniques",
        "dtype",
        "Count missing",
        "Pct. missing",
    ]
    assert overview.loc["A"].tolist() == ["1", 5, int, 0, 0.0]
    assert overview["Count uniques"].tolist() == [5, 5, 5, 4]
    assert overview["dtype"].tolist() == [int, object, object, float]
    assert overview["Count missing"].tolist() == [0, 0, 0, 1]
    assert overview["Pct. missing"].tolist() == [0.0, 0.0, 0.0, 20.0]


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
            "B": ["a", "b", "c", "d", "e"],
            "C": [3, "b", 5, "d", 7],
            "D": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    df = reduce_mem_usage(df, verbose=False)

    assert df["A"].dtype == np.int8
    assert df["B"].dtype == object
    assert df["C"].dtype == object
    assert df["D"].dtype == np.float16


def target_generate_target_bins():
    """Test generate_target_bins method"""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [3, "b", 5, "d", 7],
            "D": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    df = generate_target_bins([-1, 2, 4, np.inf], df, "A")
    assert df["A_bins"].tolist() == ["0-2", "0-2", "3-4", "3-4", "5+"]

    df = generate_target_bins([-1, 1, 3], df, "A")
    assert df["A_bins"].tolist() == ["0-1", "2-3", "2-3", np.nan, np.nan]


def test_data():
    test_data_overview()
    test_missing_data()
    test_returnNotMatches()
    test_returnMatches()
    test_reduce_mem_usage()
    target_generate_target_bins()
    print("All data tests passed! :)")


if __name__ == "__main__":
    test_data()
