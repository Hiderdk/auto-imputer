from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict
from sklearn.impute._base import BaseEstimator as BaseEstimatorImputer
import pandas as pd


class RegressionImputer(BaseEstimatorImputer):

    def __init__(self, estimator: RandomForestRegressor = LinearRegression()):
        self._estimator = estimator
        self._column_to_median: Dict[str, float] = {}
        self._feature_names: List[str] = []
        self._target_name: str = ""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._feature_names = X.columns.tolist()
        self._target_name = y.name

        for idx, column in enumerate(self._feature_names):
            median = X[column].median()
            self._column_to_median[column] = median

        X = X.fillna(value=self._column_to_median)
        self._estimator.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        is_null_indexes = df[df[self._target_name].isnull()].index.tolist()
        feature_values = df[df.index.isin(is_null_indexes)][self._feature_names]
        feature_values = feature_values.fillna(self._column_to_median)
        prediction = self._estimator.predict(feature_values)
        df.at[is_null_indexes, self._target_name] = prediction
        return df
