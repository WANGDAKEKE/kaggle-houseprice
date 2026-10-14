"""Fold-local preprocessing and models for Kaggle House Prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class Features(BaseEstimator, TransformerMixin):
    def __init__(self, engineered=False, log_numeric=False):
        self.engineered = engineered
        self.log_numeric = log_numeric

    def fit(self, X, y=None):
        frame = self._features(X)
        self.log_columns_ = []
        if self.log_numeric:
            # Learn skewed columns using this training fold only.
            for col in frame.select_dtypes(include="number"):
                values = frame[col].dropna()
                if len(values) and values.min() >= 0 and values.nunique() > 10 and values.skew() > 0.75:
                    self.log_columns_.append(col)
        return self

    def transform(self, X):
        frame = self._features(X)
        for col in self.log_columns_:
            frame[col] = np.log1p(frame[col].clip(lower=0))
        return frame

    def _features(self, X):
        x = X.copy().drop(columns=["Id", "SalePrice"], errors="ignore")
        # MSSubClass is a dwelling class code, not a continuous measurement.
        x["MSSubClass"] = x["MSSubClass"].astype(str)
        if not self.engineered:
            return x
        num = lambda c: pd.to_numeric(x[c], errors="coerce").fillna(0)
        x["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
        x["TotalFinishedSF"] = num("GrLivArea") + num("BsmtFinSF1") + num("BsmtFinSF2")
        x["TotalBath"] = num("FullBath") + 0.5 * num("HalfBath") + num("BsmtFullBath") + 0.5 * num("BsmtHalfBath")
        x["TotalPorchSF"] = sum(num(c) for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"])
        x["TotalOutdoorSF"] = x["TotalPorchSF"] + num("WoodDeckSF")
        x["HouseAge"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
        x["RemodelAge"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)
        # Incorrect future garage years are treated as unknown, not a negative age.
        garage_year = pd.to_numeric(x["GarageYrBlt"], errors="coerce")
        x["GarageAge"] = (num("YrSold") - garage_year).where(garage_year <= num("YrSold"))
        x["WasRemodeled"] = (num("YearRemodAdd") != num("YearBuilt")).astype(int)
        x["IsNew"] = (num("YearBuilt") == num("YrSold")).astype(int)
        for col in ["TotalBsmtSF", "GarageArea", "Fireplaces", "PoolArea", "2ndFlrSF", "MasVnrArea"]:
            x[f"Has_{col}"] = (num(col) > 0).astype(int)
        quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
        for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]:
            x[f"{col}_ordinal"] = x[col].map(quality).fillna(0)
        x["QualSquared"] = num("OverallQual") ** 2
        x["QualLogArea"] = num("OverallQual") * np.log1p(num("GrLivArea"))
        x["QualLogTotalSF"] = num("OverallQual") * np.log1p(x["TotalSF"])
        x["BathsPerRoom"] = x["TotalBath"] / num("TotRmsAbvGrd").clip(lower=1)
        x["AreaPerRoom"] = num("GrLivArea") / num("TotRmsAbvGrd").clip(lower=1)
        x["MonthSin"] = np.sin(2 * np.pi * num("MoSold") / 12)
        x["MonthCos"] = np.cos(2 * np.pi * num("MoSold") / 12)
        return x


class CatFrame(BaseEstimator, TransformerMixin):
    """CatBoost needs categorical missing values as strings."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categorical_ = list(X.select_dtypes(exclude="number").columns)
        self.medians_ = X.select_dtypes(include="number").median().fillna(0)
        return self

    def transform(self, X):
        x = X.copy()
        for col in self.categorical_:
            x[col] = x[col].fillna("Missing").astype(str)
        for col in self.medians_.index:
            x[col] = x[col].fillna(self.medians_[col])
        return x


class CatEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, params=None, seed=42):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            loss_function="RMSE", random_seed=self.seed, verbose=False,
            thread_count=4, allow_writing_files=False,
            cat_features=list(X.select_dtypes(exclude="number").columns),
            **(self.params or {}),
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def preprocessing(scale=True):
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=1.0)


def build_model(spec, seed=42):
    kind = spec["kind"]
    params = spec.get("params", {})
    features = Features(spec.get("engineered", False), spec.get("log_numeric", False))
    if kind == "catboost":
        return Pipeline([("features", features), ("frame", CatFrame()), ("model", CatEstimator(params, seed))])
    if kind == "ridge":
        model = Ridge(solver="lsqr", **params)
    elif kind == "lasso":
        model = Lasso(max_iter=30000, tol=1e-5, **params)
    elif kind == "elasticnet":
        model = ElasticNet(max_iter=30000, tol=1e-5, **params)
    elif kind == "randomforest":
        model = RandomForestRegressor(n_jobs=4, random_state=seed, **params)
    elif kind == "gbr":
        model = GradientBoostingRegressor(random_state=seed, **params)
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=4, random_state=seed, **params)
    elif kind == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_jobs=4, random_state=seed, verbosity=-1, **params)
    else:
        raise ValueError(kind)
    return Pipeline([("features", features), ("preprocess", preprocessing(scale=kind in ["ridge", "lasso", "elasticnet"])), ("model", model)])


def log_rmse(y, predictions):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(predictions)) ** 2)))
