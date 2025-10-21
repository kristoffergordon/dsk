from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Step:
    """
    Steps are function that are needed to run a prediction.
    Filtering of the dataset on date, data quality etc.
    are not steps but preprocessing functions.
    """

    def fit(self, df: pd.DataFrame) -> None:
        # overwrite if step is statefull
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class StandardizeScaler(Step):
    """Standardize data to zero mean and unit std."""

    col_in: list[str]
    col_out: list[str]

    def __init__(self, col_in: str | list[str], col_out: str | list[str]):
        self.col_in = [col_in] if isinstance(col_in, str) else col_in
        self.col_out = [col_out] if isinstance(col_out, str) else col_out
        self._std: pd.Series = pd.Series(dtype=float)
        self._mean: pd.Series = pd.Series(dtype=float)

    def fit(self, df: pd.DataFrame) -> None:
        # Ensure Series by operating on a DataFrame slice with a list of columns
        self._std = cast(pd.Series, df[self.col_in].std(numeric_only=True))
        self._mean = cast(pd.Series, df[self.col_in].mean(numeric_only=True))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for cin, cout in zip(self.col_in, self.col_out):
            df[cout] = (df[cin] - self._mean[cin]) / self._std[cin]
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for cin, cout in zip(self.col_in, self.col_out):
            df[cin] = df[cout] * self._std[cin] + self._mean[cin]
        return df.loc[:, self.col_in]


class ClipColumn(Step):
    col_in: str
    col_out: str
    lower: float | None
    upper: float | None

    def __init__(
        self,
        col_in: str,
        col_out: str,
        lower: float | None = None,
        upper: float | None = None,
    ):
        self.col_in = col_in
        self.col_out = col_out
        self.lower = lower
        self.upper = upper

    def transform(self, df):
        df[self.col_out] = df[self.col_in].clip(self.lower, self.upper)
        return df


class BinColumn(Step):
    def __init__(self, col: str, n_bins: int):
        self.col = col
        self.n_bins = n_bins
        self.bins = None

    def fit(self, df: pd.DataFrame) -> None:
        _, bins = pd.qcut(x=df[self.col], q=self.n_bins, retbins=True)
        self.bins = bins

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_bins = pd.cut(x=df[self.col], bins=self.bins, include_lowest=True)
        dt_bins_center = pd.Series(dt_bins).map(lambda x: x.mid)
        df[self.col] = dt_bins_center
        return df


class SelectColumns(Step):
    def __init__(self, cols: str | list[str], copy=True):
        if isinstance(cols, str):
            cols = [cols]
        self.cols = cols
        self.copy = copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[self.cols]
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if self.copy:
            result = result.copy()
        return result


class Log1pColumn(Step):
    def __init__(self, col: str):
        self.col = col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.col] = np.log1p(df[self.col])
        return df

    def inverse_transform(self, df) -> pd.DataFrame:
        df[self.col] = np.expm1(df[self.col])
        return df


class OrdinalEncodeColumns(Step):
    def __init__(self, cols: str | list[str]):
        if isinstance(cols, str):
            cols = [cols]
        self.cols = cols
        self.encoding = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.cols:
            unique_vals = df[col].unique()
            self.encoding[col] = {val: code for code, val in enumerate(unique_vals)}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            df[col] = df[col].map(self.encoding[col])
        return df

    def inverse_transform(self, df) -> pd.DataFrame:
        inverse_encoding = {
            col: {v: k for k, v in d.items()} for col, d in self.encoding.items()
        }
        for col in self.cols:
            df[col] = df[col].map(inverse_encoding[col])
        return df


class ReplaceNaNs(Step):
    def __init__(self, numeric_nan, string_nan):
        self.numeric_nan = numeric_nan
        self.string_nan = string_nan

    def transform(self, df: pd.DataFrame):
        datetime_cols = df.select_dtypes(["datetime", "timedelta"]).columns
        numeric_cols = df.select_dtypes("number").columns
        object_cols = df.select_dtypes("object").columns
        numeric_cols = list(set(numeric_cols) - set(datetime_cols))

        df[object_cols] = df[object_cols].fillna(self.string_nan)
        df[numeric_cols] = df[numeric_cols].fillna(self.numeric_nan)
        return df


class OHE_Columns(Step):
    def __init__(self, cols: str | list[str], drop_cat_cols=True):
        if isinstance(cols, str):
            cols = [cols]
        self.cols = cols
        self.OHE = OneHotEncoder(handle_unknown="ignore")
        self.OHE_columns = []
        self.drop_cat_cols = drop_cat_cols

    def fit(self, df: pd.DataFrame) -> None:
        df_cat = df[self.cols]
        self.OHE.fit(df_cat)
        self.OHE_columns = self.OHE.get_feature_names_out(
            df_cat.columns
        )  # get_feature_names changed to get_feature_names_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ohe_array = self.OHE.transform(df[self.cols]).toarray()
        df[self.OHE_columns] = ohe_array

        if self.drop_cat_cols:
            df = df.drop(columns=self.cols)

        return df

    def inverse_transform(self, df) -> pd.DataFrame:
        df[self.cols] = self.OHE.inverse_transform(df[self.OHE_columns])
        return df
