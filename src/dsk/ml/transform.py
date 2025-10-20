from typing import List

import pandas as pd

from dsk.ml.step import Step


class Sequential:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def fit(self, df: pd.DataFrame) -> None:
        for step in self.steps:
            step.fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.transform(df)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in reversed(self.steps):
            df = step.inverse_transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.fit_transform(df)
        return df
