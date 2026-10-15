"""v3 feature bases, rare-category scaling, splines and additional tree losses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from src.house_prices import Features, CatFrame
from src.house_prices_v2 import DomainFeatures


class AdvancedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, power="log", power_lambda=.15, interactions="none", quality_categories=False):
        self.power = power
        self.power_lambda = power_lambda
        self.interactions = interactions
        self.quality_categories = quality_categories

    def fit(self, X, y=None):
        self.base_ = DomainFeatures(rich=True, log_numeric=False).fit(X)
        frame = self._frame(X)
        self.power_columns_ = []
        for col in frame.select_dtypes(include="number"):
            values = frame[col].dropna()
            if len(values) and values.min() >= 0 and values.nunique() > 10 and values.skew() > .75:
                self.power_columns_.append(col)
        return self

    def transform(self, X):
        frame = self._frame(X)
        for col in self.power_columns_:
            values = frame[col].clip(lower=0)
            frame[col] = boxcox1p(values, self.power_lambda) if self.power == "boxcox" else np.log1p(values)
        return frame

    def _frame(self, X):
        frame = self.base_.transform(X)
        if self.quality_categories:
            for col in ["OverallQual", "OverallCond"]:
                frame[col + "Category"] = X[col].astype(str)
        if self.interactions in ["sale", "both"]:
            frame["SaleConditionQual"] = X.SaleCondition.astype(str) + "_" + X.OverallQual.astype(str)
            frame["SaleConditionType"] = X.SaleCondition.astype(str) + "_" + X.SaleType.fillna("Unknown").astype(str)
            frame["ConditionAge"] = X.OverallCond * frame.HouseAge
        if self.interactions in ["size", "both"]:
            # Generic physical size/era bins, shared by every neighborhood.
            size = pd.cut(X.GrLivArea, [-np.inf, 750, 1000, 1500, 2000, 2500, 3000, 4000, np.inf], labels=False).astype(str)
            era = pd.cut(X.YearBuilt, [-np.inf, 1940, 1960, 1980, 2000, np.inf], labels=False).astype(str)
            frame["NeighborhoodSize"] = X.Neighborhood.astype(str) + "_" + size
            frame["NeighborhoodEra"] = X.Neighborhood.astype(str) + "_" + era
            frame["DwellingStyle"] = X.MSSubClass.astype(str) + "_" + X.HouseStyle.astype(str)
        return frame


class CategoryScale(BaseEstimator, TransformerMixin):
    def __init__(self, power=.25):
        self.power = power

    def fit(self, X, y=None):
        p = np.asarray(X.mean(axis=0)).ravel()
        self.scale_ = np.maximum(p * (1 - p), 1e-4) ** (-self.power)
        return self

    def transform(self, X):
        return X.multiply(self.scale_).tocsr() if hasattr(X, "multiply") else X * self.scale_


def preprocessing(spec):
    categorical = [("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
                   ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=spec.get("min_frequency"), sparse_output=True))]
    if spec.get("category_scale", 0):
        categorical.append(("scale", CategoryScale(spec["category_scale"])))
    columns = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), make_column_selector(dtype_include=np.number)),
        ("categorical", Pipeline(categorical), make_column_selector(dtype_exclude=np.number)),
    ]
    if spec.get("splines", False):
        columns.append(("splines", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("basis", SplineTransformer(n_knots=spec.get("n_knots", 4), degree=2, knots="quantile", extrapolation="linear", include_bias=False, sparse_output=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), ["GrLivArea", "TotalSF", "HouseAge", "RemodelAge", "OverallQual", "OverallCond", "LotArea", "GarageArea", "TotalBath", "QualitySum"]))
    return ColumnTransformer(columns, sparse_threshold=1.0)


class FlexibleCat(RegressorMixin, BaseEstimator):
    def __init__(self, params=None, seed=42):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        params = {"loss_function": "RMSE", **(self.params or {})}
        self.model_ = CatBoostRegressor(random_seed=self.seed, thread_count=4, verbose=False,
            allow_writing_files=False, cat_features=list(X.select_dtypes(exclude="number").columns), **params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class HistogramFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.select_dtypes(exclude="number")}
        return self

    def transform(self, X):
        frame = X.copy()
        for col, categories in self.categories_.items():
            frame[col] = pd.Categorical(frame[col], categories=categories)
        return frame


def build_model(spec, seed=42):
    if spec.get("reference"):
        from src.house_prices_v2 import build_v2_model
        return build_v2_model(spec["reference_spec"], seed)
    features = (Features(engineered=True, log_numeric=False) if spec.get("features") == "v1" else
                AdvancedFeatures(power=spec.get("power", "log"), power_lambda=spec.get("power_lambda", .15),
                    interactions=spec.get("interactions", "none"), quality_categories=spec.get("quality_categories", False)))
    params = spec.get("params", {})
    if spec["kind"] == "catboost":
        return Pipeline([("features", features), ("frame", CatFrame()), ("model", FlexibleCat(params, seed))])
    if spec["kind"] == "histgb":
        model = HistGradientBoostingRegressor(categorical_features="from_dtype", early_stopping=False, random_state=seed, **params)
        return Pipeline([("features", features), ("frame", HistogramFrame()), ("model", model)])
    if spec["kind"] == "elasticnet":
        model = ElasticNet(max_iter=30000, tol=1e-5, **params)
    elif spec["kind"] == "ridge":
        model = Ridge(solver="lsqr", **params)
    elif spec["kind"] == "gbr":
        model = GradientBoostingRegressor(random_state=seed, **params)
    else:
        raise ValueError(spec["kind"])
    return Pipeline([("features", features), ("preprocess", preprocessing(spec)), ("model", model)])
