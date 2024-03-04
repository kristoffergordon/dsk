from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy import stats

sns.set_theme()


def correlation_heatmap(df: pd.DataFrame, h=10, w=10, cmap="coolwarm"):
    """Method for plotting a correlation heatmap

    Args:
        df : Pandas dataframe
            Dataframe to plot correlations for
        h : int
            Height of plot
        w : int
            Width of plot
        cmap : seaborn Colormap
            Colomarp of the correlation heatmap

    Returns:
        None
    """
    df = df.select_dtypes("number")
    correlations = df.corr()

    _, _ = plt.subplots(figsize=(h, w))
    sns.heatmap(
        correlations,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.80},
    )
    plt.show()


def heatmap(correlations, h=10, w=10, cmap="coolwarm"):
    """Method for plotting a heatmap of the correlations between features

    Args:
        correlations : array-like
        h : int
            Height of plot
        w : int
            Width of plot
        cmap : seaborn Colormap
            Colomarp of the correlation heatmap

    Returns:
        None
    """

    _, _ = plt.subplots(figsize=(h, w))
    sns.heatmap(
        correlations,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.80},
    )
    plt.show()


# histogram and normal probability plot
def probplts(y):
    _, (ax_top, _) = plt.subplots(2, figsize=(6, 11))
    plt.subplots_adjust(hspace=0.4)
    sns.histplot(y, kde=True, ax=ax_top)
    ax_top.set_title("Distribution plot")

    ax_top.axvline(y.mean(), color="r", linestyle="--", label="Mean")
    ax_top.axvline(y.median(), color="g", linestyle="-", label="Median")
    ax_top.legend()

    stats.probplot(y, plot=plt)

    plt.show()


def pred_error_plt(y_pred, y_test, rng_min=0, rng_max=200):
    """
    Plots the predictions alongside the true targets

    Args:
        y_pred: List of model target predictions
        y_test: True targets
        rng_min: Start of x-range plot
        rng_max: End of x.range plot

    """
    error = y_pred - y_test

    # Initialize figure
    _, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(15, 7), gridspec_kw={"height_ratios": [1.4, 1]}
    )

    # Plot targets and predictions
    ax1.plot(y_test[rng_min:rng_max], label="True")
    ax1.plot(y_pred[rng_min:rng_max], label="Predicted")
    ax1.legend()
    ax1.set_ylabel("Dwell time [days]")

    # Plot corresponding residual
    ax2.axhline(y=0, xmin=rng_min, xmax=rng_max, color="r", linestyle="-")
    ax2.plot(error[rng_min:rng_max])
    ax2.fill_between(range(0, len(error[rng_min:rng_max])), 0, error[rng_min:rng_max])
    ax2.set_xlabel("Datapoint")
    ax2.set_ylabel("Error [days]")

    plt.show()
    return plt


def serial_boxplot(
    df: pd.DataFrame,
    y: str,
    columns: List[str] = None,
    upper: int = 25,
    lower: int = 2,
    h: int = 20,
    w: int = 5,
) -> None:
    """Draw a box plot to show distributions of y with respect to columns

    Args:
        df : pandas.DataFrame
            Dataframe with data to be plottet in boxplot
        y : str
            Target column to be used as reference in boxplot. Has to be numerical
        columns : list
            List of columns that y is plottet against. All object columnes will be
            used if no list is provided
        upper : int
            Upper limit for unique count. Columns with a unique count larger
            than upper limit are skipped
        lower : int
            Lower limit for unique count. Columns with a unique count lower
            than lower limit are skipped
        h : int
            Height of boxplot figure
        w : int
            Width of bowplot figure

    Returns:
        plt : matplotlib.pyplot
            Boxplot of y with respect to columns

    Raises:
        TypeError: Ensure that df is a pandas.DataFrame
        TypeError: Ensure that y is a numerical column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df has to be a DataFrame at not {type(df)}")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y has to be a numeric column at not {df[y].dtype}")

    # Boxplot all columns if no list is provided
    if not (columns):
        columns = list(df.select_dtypes("object").columns)

    # Check if y is in list of columns
    if y in columns:
        columns.remove(y)

    # Allow single column to be passed as string
    if isinstance(columns, str):
        columns = [columns]

    # Iterate over columns
    for col in columns:
        # Skip columns outside upper/lower limit.
        if df[col].nunique() >= upper or df[col].nunique() < lower:
            continue

        # Calculate number of obs per group & median to position labels
        medians = df.groupby([col])[y].median()
        order = df[col].value_counts().index
        nobs = df[col].value_counts().values
        nobs = [str(np.round(x / len(df) * 100, 1)) for x in nobs]

        # Draw boxplot and order order by decreasing number of obs
        plt.figure(figsize=(h, w))
        ax = sns.boxplot(data=df, x=col, y=y, hue=col, order=order, palette="tab10")
        plt.xticks(rotation=45)

        # Add nobs to the plot above median
        pos = range(len(nobs))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(
                pos[tick],
                medians[label.get_text()] + 0.05 * medians[label.get_text()],
                nobs[tick],
                horizontalalignment="center",
                size="medium",
                color="w",
                weight="semibold",
            )

        plt.show()

    return None
