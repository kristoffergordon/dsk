from unittest.mock import patch

import numpy as np
import pandas as pd

from dsk import plots

rng = np.random.default_rng(2021)
df = pd.DataFrame(rng.random((10, 5)), columns=["A", "B", "C", "D", "E"])


@patch("matplotlib.pyplot.show")
def test_plot_fn(mock_show):
    plots.correlation_heatmap(df)


@patch("matplotlib.pyplot.show")
def test_heatmap(mock_show):
    plots.heatmap(df.corr())


@patch("matplotlib.pyplot.show")
def test_probsplts(mock_show):
    s = pd.Series(rng.standard_normal(1000))
    _ = plots.probplots(s)


@patch("matplotlib.pyplot.show")
def test_probsplots(mock_show):
    y_test = rng.random(100)
    y_pred = y_test + rng.normal(0, 0.2, 100)
    plots.pred_error_plt(y_pred, y_test, rng_min=0, rng_max=100)


@patch("matplotlib.pyplot.show")
def test_serial_boxplot(mock_show):
    # Create a DataFrame with some random data using the generator object
    df = pd.DataFrame(
        {
            "A": rng.choice(["Very Low", "Low", "Medium", "High", "Very High"], 250),
            "B": rng.choice(["Yes", "No", "Maybe"], 250),
            "C": rng.random(250),
        }
    )

    plots.serial_boxplot(df, "C")
