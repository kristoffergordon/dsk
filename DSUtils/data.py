import numpy as np
import pandas as pd


def dataOverview(df, sort_by_missing=False):
    # Preliminary data overview
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


def missingData(df):
    # missing data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(
        ascending=False
    ) * 100
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    return missing_data


def returnNotMatches(a, b):
    return [x for x in b if x not in a]


def returnMatches(a, b):
    return [x for x in b if x in a]


def reduce_mem_usage(df, verbose=True):
    """Simple method for reducing memory usage of a dataframe"""
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
