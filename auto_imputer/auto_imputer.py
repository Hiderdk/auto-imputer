from typing import List, Literal, Dict
from models import ColumnData
import pandas as pd
from sklearn.impute._base import BaseEstimator as BaseEstimatorImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.linear_model import LogisticRegression
from regression_imputer import RegressionImputer


class AutoImputer(BaseEstimatorImputer):

    def __init__(self,
                 missing_value_to_drop_threshold: float = 0.6,
                 estimator: LogisticRegression = RandomForestClassifier(n_estimators=10, max_depth=10,
                                                                        min_samples_split=4),
                 scorer: roc_auc_score = roc_auc_score,
                 min_score: float = 0.536, statistical_significance: float = 0.05, min_correlation: float = 0.7,
                 regression_imputer: RegressionImputer = RegressionImputer(),
                 default_impute_strategy: Literal["mean", "median"] = "median"):

        self._missing_value_to_drop_threshold: float = missing_value_to_drop_threshold
        self._estimator: RandomForestClassifier = estimator
        self._scorer: roc_auc_score = scorer
        self._min_score: float = min_score
        self._columns_data: Dict[str, ColumnData] = {}
        self._statistical_significance = statistical_significance
        self._min_correlation: float = min_correlation
        self._regression_imputer: RegressionImputer = regression_imputer
        self._default_impute_strategy: Literal["mean", "median"] = default_impute_strategy

    def fit(self, X: pd.DataFrame, y: pd.Series):
        new_df = X.copy().reset_index()
        new_df["target"] = y.tolist()

        indexes = new_df.index.tolist()
        feature_to_is_null_indexes = self._generate_is_null_indexes(new_df)

        for column_name in X.columns:
            column_data = self._fit_single_column(new_df, feature_to_is_null_indexes)
            self._columns_data[column_name] = column_data

    def _fit_single_column(self, new_df: pd.DataFrame, feature_to_is_null_indexes: Dict[str, List[int]],
                           indexes: List[int], column_name: str) -> ColumnData:
        is_null_indexes = feature_to_is_null_indexes[column_name]
        not_null_indexes = list(set(indexes) - set(is_null_indexes))
        not_null_df = new_df[new_df.index.isin(not_null_indexes)]
        percent_missing = len(is_null_indexes) / len(indexes)

        mean = new_df[column_name].mean()
        median = new_df[column_name].median()

        if percent_missing == 0:
            return ColumnData(percent_missing=percent_missing,
                              mean=mean,
                              median=median,
                              impute_method="simple_imputer",
                              impute_strategy=self._default_impute_strategy
                              )

        score = self._calculate_score_with_target(new_df=new_df, is_null_indexes=is_null_indexes,
                                                  column_name=column_name)

        is_normally_distributed = self._is_normally_distributed(values=not_null_df[column_name])
        other_feature_correlation, correlated_features = \
            self._generate_correlated_features(column_name=column_name, X=new_df,
                                               column_to_is_null_indexes=feature_to_is_null_indexes)

        impute_method = self._get_method(percent_missing=percent_missing, score=score,
                                         other_feature_correlation=other_feature_correlation)
        impute_strategy = self._get_impute_strategy(normally_distributed=is_normally_distributed,
                                                    impute_method=impute_method)

        add_is_missing = self._get_add_is_missing(score)
        if other_feature_correlation:

            regression_imputer = self._generate_regression_imputer(not_null_df=not_null_df,
                                                                   correlated_features=correlated_features,
                                                                   column_name=column_name)
            return ColumnData(percent_missing=percent_missing, score=score, impute_method=impute_method,
                              impute_strategy=impute_strategy, added_is_missing_feature=add_is_missing,
                              mean=mean, median=mean, impute_estimator=regression_imputer)

        else:
            return ColumnData(percent_missing=percent_missing, score=score, impute_method=impute_method,
                              impute_strategy=impute_strategy, added_is_missing_feature=add_is_missing,
                              mean=mean, median=mean)

    def _generate_regression_imputer(self, not_null_df: pd.DataFrame, correlated_features: List[str],
                                     column_name: str) -> RegressionImputer:
        regression_imputer: RegressionImputer = RegressionImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=4))
        regression_imputer.fit(not_null_df[correlated_features], not_null_df[column_name])
        return regression_imputer

    def _get_add_is_missing(self, score: float) -> bool:
        add_is_missing = False
        if score > self._min_score:
            add_is_missing = True
        return add_is_missing

    def _calculate_score_with_target(self, new_df: pd.DataFrame, is_null_indexes: List[int], column_name: str) -> float:
        new_df[column_name + '_is_null'] = 0
        new_df.at[is_null_indexes, column_name + '_is_null'] = 1
        estimator: RandomForestClassifier = clone(self._estimator)
        estimator.fit(new_df[[column_name + '_is_null']], new_df["target"])
        prob = estimator.predict_proba(new_df[[column_name + '_is_null']])[:, 1]
        score = self._scorer(new_df["target"], prob)
        return score

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_df = X.copy()
        drop_count = 0
        for column in X.columns:

            if self._columns_data[column].impute_method == "drop":
                new_df.drop(columns=[column])
                drop_count += 1

            elif self._columns_data[column].added_is_missing_feature:
                new_df[column + '_is_missing'] = 0
                new_df.loc[new_df[column].isnull(), column + '_is_missing'] = 1

            if self._columns_data[column].impute_strategy == "regression":
                new_df = self._columns_data[column].impute_estimator.transform(new_df)

            elif self._columns_data[column].impute_strategy == "median":
                new_df[column] = new_df[column].fillna(self._columns_data[column].median)

            elif self._columns_data[column].impute_strategy == "mean":
                new_df[column] = new_df[column].fillna(self._columns_data[column].mean)
        print("drop count", drop_count)
        return new_df

    def _generate_is_null_indexes(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        feature_to_is_null_indexes: Dict[str, List[int]] = {}
        for column in df.columns:
            is_null_indexes = df[df[column].isnull()].index.tolist()
            feature_to_is_null_indexes[column] = is_null_indexes

        return feature_to_is_null_indexes

    def _generate_correlated_features(self, column_name: str, X: pd.DataFrame,
                                      column_to_is_null_indexes: Dict[str, List[int]]) -> [bool,
                                                                                           List[str]]:
        is_null_indexes = column_to_is_null_indexes[column_name]
        correlated_features: List[str] = []
        new_X = X.copy()
        new_X = new_X[new_X[column_name].notna()]
        is_correlated_with_other_features: bool = False
        for column in new_X.columns:
            if column != column_name:

                other_column_is_null_indexes = column_to_is_null_indexes[column]
                same_is_null_indexes = list(set(is_null_indexes).intersection(other_column_is_null_indexes))
                if len(same_is_null_indexes) / len(is_null_indexes) > 0.8:
                    continue

                abs_corr = abs(X[column_name].corr(X[column]))
                if abs_corr > self._min_correlation:
                    correlated_features.append(column)
                    is_correlated_with_other_features = True

        return is_correlated_with_other_features, correlated_features

    def _get_method(self, percent_missing: float, score: float, other_feature_correlation: bool) -> \
            Literal["drop", "simple_imputer", "predict"]:
        if score < self._min_score and percent_missing > self._missing_value_to_drop_threshold:
            return "drop"
        elif other_feature_correlation:
            return "predict"

        return "simple_imputer"

    def _is_normally_distributed(self, values: pd.Series) -> bool:
        k2, p = stats.normaltest(values)
        if p < self._statistical_significance:
            return False

        return True

    def _get_impute_strategy(self, normally_distributed: bool, impute_method: Literal["simple_imputer", "predict"]) -> \
            Literal["mean", "median", "regression", "None"]:
        if normally_distributed and impute_method == "simple_imputer":
            return "mean"
        elif not normally_distributed and impute_method == "simple_imputer":
            return "median"
        elif impute_method == "predict":
            return "regression"

        return "None"
