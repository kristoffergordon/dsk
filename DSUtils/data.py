from typing import List

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def data_overview(df, sort_by_missing=False):
    """Method for getting an preliminary data overview

    Args:
        df : Pandas dataframe
            Dataframe to get an overview of
        sort_by_missing : bool
            Sort the overview by missing values

    Returns:
        overview_df : Pandas dataframe
            Dataframe with an overview of the data

    Raises:
        TypeError: If df is not a pandas dataframe
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    uniques = df.nunique()
    dtypes = df.dtypes
    total = df.isnull().sum().sort_values()
    percent = (df.isnull().sum() / df.isnull().count()).sort_values() * 100

    # Try to get a value sample from each column
    if len(df.dropna()) > 0:
        sample = df.dropna().iloc[0].astype(str).apply(lambda x: x[:30])
    else:
        sample = df.loc[0].astype(str).apply(lambda x: x[:30])

    data_overview = [sample, uniques, dtypes, total, percent]
    keys = ["Sample", "Count uniques", "dtype", "Count missing", "Pct. missing"]

    overview_df = pd.concat(data_overview, keys=keys, axis=1, sort=False)
    if sort_by_missing:
        overview_df = overview_df.sort_values(by="Pct. missing", ascending=False)
    overview_df = overview_df.round(1)
    return overview_df


def missing_data(df):
    """Method for getting an overview of missing data

    Args:
        df : Pandas dataframe
            Dataframe to get an overview of

    returns:
        missing_data : Pandas dataframe
            Dataframe with an overview of missing data
    """

    # missing data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = df.isnull().sum() / df.isnull().count()
    percent = percent.sort_values(ascending=False) * 100
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    return missing_data


def return_not_matches(a, b):
    """Method for returning values in b that are not in a"""
    return [x for x in b if x not in a]


def return_matches(a, b):
    """Method for returning values in b that are also in a"""
    return [x for x in b if x in a]


def reduce_mem_usage(df, verbose=True):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    start_mem = df.memory_usage().sum() / 1024**2
    df = convert_df(df)
    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        med_red = 100 * (start_mem - end_mem) / start_mem
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({med_red:.1f}% reduction)")

    return df


def convert_df(df):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            df[col] = convert_col(df[col], col_type)
    return df


def convert_col(col, col_type):
    c_min = col.min()
    c_max = col.max()
    if str(col_type).startswith("int"):
        return convert_int_col(col, c_min, c_max)
    else:
        return convert_float_col(col, c_min, c_max)


def convert_int_col(col, c_min, c_max):
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        return col.astype(np.int8)
    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        return col.astype(np.int16)
    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        return col.astype(np.int32)
    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        return col.astype(np.int64)
    else:
        return col


def convert_float_col(col, c_min, c_max):
    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        return col.astype(np.float16)
    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
        return col.astype(np.float32)
    else:
        return col.astype(np.float64)


def generate_target_bins(bins: List[int], df: pd.DataFrame, target: str):
    """Method for generating a set of bins for the target column

    Args:
        bins : List of integers
            List of range between target bins. The target bins will span
            the gap between then entries in this list and will always start at n+1.
            Therefore, to start the class at 0, one should let the first entry in the
            list be -1. See Example usage for example.
        df : pd.DataFrame
            Dataframe with terminal data. Must contain target column
        target : String
            Name of the target column to be binned

    Returns:
        df_copy: Copy of df with target bins

    Raises:
        KeyError: if the target is not in df
        TypeError: if bins is not a list or if target is not numeric
        ValueError: if bins is not increasing monotonically

    Example usage:
        ### With np.inf as final class ###
        Input bins: [-1, 3, 7, 11, 15, np.inf]
        Output labels: ['0-3', '4-7', '8-11', '12-15', '16+']

        ### With finite int as final class ###
        Input bins: [-1, 6, 13, 20, 27]
        Output labels: ['0-6', '7-13', '14-20', '21-27']
    """

    # Ensure that target column exists in df
    if target not in df.columns:
        raise KeyError(f"Target column {target} not in DataFrame")

    if not isinstance(bins, list):
        raise TypeError("bins must be a list")

    if not ptypes.is_numeric_dtype(df[target]):
        raise TypeError("target must be numeric")

    # Ensure that bins are increasing monotonically
    if sorted(bins) != bins:
        raise ValueError("bins must increase monotonically")

    # Copy dataframe to avoid overwriting
    df_copy = df.copy(deep=True)

    # Generate class names based on bins
    n_bins = len(bins) - 1
    labels = [f"{b1+1}-{b2}" for b1, b2 in zip(bins[0 : (n_bins - 1)], bins[1:n_bins])]

    # Check if final class is infinity -> then add last class+ as name
    if bins[-1] == np.inf:
        labels = labels + [f"{bins[-2]+1}+"]
    else:
        labels = labels + [f"{bins[-2]+1}-{bins[-1]}"]

    # Add target bins to dataframe (copy)
    target_bins = f"{target}_bins"
    df_copy[target_bins] = pd.cut(df_copy[target], bins, labels=labels)

    return df_copy
