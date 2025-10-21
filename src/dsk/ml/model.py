from __future__ import annotations

import glob
import os
import pathlib
import pickle
from datetime import datetime

import pandas as pd


class MLModel:
    """
    Model container class

    save_model will save to 'folder/MODELNAME/TIMESTAMP/model.pkl'
    save_dataset will save to 'folder/MODELNAME_TIMESTAMP.parquet'

    where model_folder and dataset folder are given as arguments
    """

    model_name = None

    def __init__(self):
        if self.model_name is None:
            raise ValueError

    def create_train_data(self, **kwargs) -> pd.DataFrame:
        # preprocessing e.g. filtering on date and data quality
        raise NotImplementedError

    def fit(self, df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def predict_live(self, raw_df):
        raise NotImplementedError

    def save_model(self, folder) -> pathlib.Path:
        fn = self._get_model_name(folder, self._get_timestr())
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "wb") as fh:
            pickle.dump(self, fh)
        return pathlib.Path(fn)

    def save_dataset(self, df: pd.DataFrame, folder) -> pathlib.Path:
        fn = self._get_dataset_name(folder, self._get_timestr())
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        df.to_parquet(fn)
        return pathlib.Path(fn)

    @classmethod
    def load_newest_model(cls, folder) -> tuple[MLModel, pathlib.Path]:
        fns = cls._saved_models(folder)
        if not fns:
            raise ValueError("No models found")
        with open(fns[-1], "rb") as fh:
            model = pickle.load(fh)
        return model, fns[-1]

    @classmethod
    def load_newest_dataset(cls, folder) -> tuple[pd.DataFrame, pathlib.Path]:
        fns = cls._saved_datasets(folder)
        if not fns:
            raise ValueError("No models found")
        df: pd.DataFrame = pd.read_parquet(fns[-1])
        return df, fns[-1]

    @classmethod
    def _get_model_name(cls, model, subfolder):
        if cls.model_name is None:
            raise ValueError("model_name is not set")
        return os.path.join(model, "models", cls.model_name, subfolder, "model.pkl")

    @classmethod
    def _get_dataset_name(cls, folder, postfix):
        return os.path.join(folder, "datasets", f"{cls.model_name}_{postfix}.parquet")

    @classmethod
    def _saved_models(cls, folder):
        pattern = cls._get_model_name(folder, "*")
        return list(sorted([pathlib.Path(fn) for fn in glob.glob(pattern)]))

    @classmethod
    def _saved_datasets(cls, folder):
        pattern = cls._get_dataset_name(folder, "*")
        return list(sorted([pathlib.Path(fn) for fn in glob.glob(pattern)]))

    @staticmethod
    def _get_timestr() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")
