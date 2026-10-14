"""Additional fold-local feature treatment and kernel models for v2 experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from src.house_prices import Features, CatFrame, CatEstimator


class DomainFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, rich=False, log_numeric=True, cap_quantile=None):
        self.rich = rich
        self.log_numeric = log_numeric
        self.cap_quantile = cap_quantile

    def fit(self, X, y=None):
        x = X.copy()
        self.frontage_by_neighborhood_ = x.groupby("Neighborhood", observed=True).LotFrontage.median().to_dict()
        self.frontage_median_ = x.LotFrontage.median()
        self.modes_ = {}
        for col in ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]:
            mode = x[col].mode()
            self.modes_[col] = mode.iloc[0] if len(mode) else "Unknown"
        frame = self._features(x)
        self.caps_ = {}
        if self.cap_quantile is not None:
            for col in ["LotArea", "LotFrontage", "GrLivArea", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "MasVnrArea", "TotalSF", "TotalFinishedSF", "TotalOutdoorSF", "AreaPerRoom"]:
                if col in frame:
                    self.caps_[col] = float(frame[col].quantile(self.cap_quantile))
        frame = self._cap(frame)
        self.log_columns_ = []
        if self.log_numeric:
            for col in frame.select_dtypes(include="number"):
                values = frame[col].dropna()
                if len(values) and values.min() >= 0 and values.nunique() > 10 and values.skew() > 0.75:
                    self.log_columns_.append(col)
        return self

    def transform(self, X):
        frame = self._cap(self._features(X))
        for col in self.log_columns_:
            frame[col] = np.log1p(frame[col].clip(lower=0))
        return frame

    def _cap(self, frame):
        for col, maximum in self.caps_.items():
            frame[col] = frame[col].clip(upper=maximum)
        return frame

    def _features(self, X):
        x = X.copy()
        # Treat documented absence of a facility separately from unknown values.
        for col in ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]:
            x[col] = x[col].fillna("None")
        for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea", "MasVnrArea"]:
            x[col] = x[col].fillna(0)
        x["LotFrontage"] = x.LotFrontage.fillna(x.Neighborhood.map(self.frontage_by_neighborhood_)).fillna(self.frontage_median_)
        x["Functional"] = x.Functional.fillna("Typ")
        for col, value in self.modes_.items():
            x[col] = x[col].fillna(value)
        # Masonry type genuinely unknown with positive area stays unknown.
        x["MasVnrType"] = x.MasVnrType.fillna(x.MasVnrArea.map(lambda area: "None" if area == 0 else "Unknown"))
        x["GarageYrBlt"] = x.GarageYrBlt.where(x.GarageYrBlt <= x.YrSold)
        has_garage = x.GarageCars > 0
        x.loc[has_garage, "GarageYrBlt"] = x.loc[has_garage, "GarageYrBlt"].fillna(x.loc[has_garage, "YearBuilt"])
        x.loc[~has_garage, "GarageYrBlt"] = x.loc[~has_garage, "YearBuilt"]
        frame = Features(engineered=True)._features(x)
        frame["MoSold"] = frame.MoSold.astype(str)
        frame["YrSoldCategory"] = frame.YrSold.astype(str)
        frame["BsmtExposure_ordinal"] = x.BsmtExposure.map({"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).fillna(0)
        for col in ["BsmtFinType1", "BsmtFinType2"]:
            frame[col + "_ordinal"] = x[col].map({"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}).fillna(0)
        frame["Functional_ordinal"] = x.Functional.map({"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}).fillna(7)
        frame["GarageFinish_ordinal"] = x.GarageFinish.map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}).fillna(0)
        frame["BsmtFinishedRatio"] = (x.BsmtFinSF1 + x.BsmtFinSF2) / x.TotalBsmtSF.clip(lower=1)
        frame["GarageAreaPerCar"] = x.GarageArea / x.GarageCars.clip(lower=1)
        if self.rich:
            frame["OverallQualCond"] = x.OverallQual * x.OverallCond
            frame["QualitySum"] = x.OverallQual + frame.ExterQual_ordinal + frame.KitchenQual_ordinal + frame.BsmtQual_ordinal
            frame["QualityLogFinishedArea"] = frame.QualitySum * np.log1p(frame.TotalFinishedSF)
            frame["KitchenQualLogArea"] = frame.KitchenQual_ordinal * np.log1p(x.GrLivArea)
            frame["ExterQualLogArea"] = frame.ExterQual_ordinal * np.log1p(x.GrLivArea)
            frame["QualityAge"] = x.OverallQual * frame.HouseAge
            frame["BasementFraction"] = x.TotalBsmtSF / frame.TotalSF.clip(lower=1)
            frame["SecondFloorFraction"] = x["2ndFlrSF"] / x.GrLivArea.clip(lower=1)
            frame["LotToLivingRatio"] = x.LotArea / x.GrLivArea.clip(lower=1)
            frame["HasMultipleConditions"] = (x.Condition2 != "Norm").astype(int)
            frame["NeighborhoodQual"] = x.Neighborhood.astype(str) + "_" + x.OverallQual.astype(str)
        return frame


def v2_preprocessing(scale="standard", min_frequency=None, dense=False):
    scaler = RobustScaler() if scale == "robust" else StandardScaler()
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale is not None:
        numeric.append(("scale", scaler))
    return ColumnTransformer([
        ("num", Pipeline(numeric), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=not dense)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=0.0 if dense else 1.0)


def build_v2_model(spec, seed=42):
    if spec.get("features", "v2") == "v1":
        from src.house_prices import build_model
        return build_model(spec, seed)
    features = DomainFeatures(rich=spec.get("rich", False), log_numeric=spec.get("log_numeric", True), cap_quantile=spec.get("cap_quantile"))
    kind, params = spec["kind"], spec.get("params", {})
    if kind == "catboost":
        return Pipeline([("features", features), ("frame", CatFrame()), ("model", CatEstimator(params, seed))])
    if kind == "ridge":
        model = Ridge(solver="lsqr", **params)
    elif kind == "lasso":
        model = Lasso(max_iter=30000, tol=1e-5, **params)
    elif kind == "elasticnet":
        model = ElasticNet(max_iter=30000, tol=1e-5, **params)
    elif kind == "svr":
        model = SVR(**params)
    elif kind == "kernelridge":
        model = KernelRidge(**params)
    elif kind == "huber":
        model = HuberRegressor(max_iter=1000, **params)
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=4, random_state=seed, **params)
    elif kind == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_jobs=4, random_state=seed, verbosity=-1, **params)
    else:
        raise ValueError(kind)
    scale = spec.get("scale", "standard") if kind not in ["xgboost", "lightgbm"] else None
    return Pipeline([("features", features), ("preprocess", v2_preprocessing(scale, spec.get("min_frequency"), spec.get("dense", False))), ("model", model)])


def training_weights(X, y_log, scheme):
    if scheme == "none":
        return None
    weights = np.ones(len(X))
    # Labels used only inside each fit; every validation row is still evaluated.
    large_low_price = (X.GrLivArea.to_numpy() > 4000) & (y_log < np.log(300000))
    if scheme == "downweight_large":
        weights[large_low_price] = 0.1
    else:
        raise ValueError(scheme)
    return weights
