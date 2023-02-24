from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import learning_curve


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

    fig, ax = plt.subplots(figsize=(h, w))
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

    fig, ax = plt.subplots(figsize=(h, w))
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
    f, (ax_top, ax_bot) = plt.subplots(2, figsize=(6, 11))
    plt.subplots_adjust(hspace=0.4)
    sns.distplot(y, fit=norm, ax=ax_top)
    ax_top.set_title("Distribution plot")

    ax_top.axvline(y.mean(), color="r", linestyle="--", label="Mean")
    ax_top.axvline(y.median(), color="g", linestyle="-", label="Median")
    ax_top.legend()

    # fig = plt.figure()
    res = stats.probplot(y, plot=plt)

    plt.show()
    return plt


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
    fig, (ax1, ax2) = plt.subplots(
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


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Call learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
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
        ax = sns.boxplot(data=df, x=col, y=y, order=order)
        plt.xticks(rotation=45)

        # Add nobs to the plot above median
        pos = range(len(nobs))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(
                pos[tick],
                medians[label.get_text()] + 0.2,
                nobs[tick],
                horizontalalignment="center",
                size="medium",
                color="w",
                weight="semibold",
            )

        plt.show()
