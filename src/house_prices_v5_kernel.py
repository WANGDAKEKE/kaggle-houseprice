"""Small, fold-local kernel/linear search with compact physical features.

No target-dependent row removal or external observations are used. Kernel
targets are centered inside each training fit so the intercept is learned.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVR

from src.house_prices_v2 import DomainFeatures
from src.house_prices_v3 import CategoryScale


class CompactPhysicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_only=True, ages_only=True):
        self.ordinal_only = ordinal_only
        self.ages_only = ages_only

    def fit(self, X, y=None):
        self.base_ = DomainFeatures(rich=True, log_numeric=True).fit(X)
        return self

    def transform(self, X):
        frame = self.base_.transform(X)
        # Sparse neighborhood-by-quality crosses were one source of instability
        # in previous rounds. Keep neighborhood and numeric quality separately.
        drop = ["NeighborhoodQual", "YrSoldCategory"]
        if self.ordinal_only:
            drop += ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                     "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
                     "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Functional", "GarageFinish"]
        if self.ages_only:
            drop += ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
        return frame.drop(columns=drop, errors="ignore")


def build_model(spec, seed=42):
    """Create an unfitted sklearn model; all transforms learn on fit rows only."""
    scaler = RobustScaler(quantile_range=(10., 90.)) if spec.get("scale") == "robust" else StandardScaler()
    cat = [("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
           ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=spec.get("min_frequency", 3), sparse_output=True))]
    if spec.get("category_scale", 0):
        cat.append(("scale", CategoryScale(power=spec["category_scale"])))
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", scaler)]), make_column_selector(dtype_include=np.number)),
        ("category", Pipeline(cat), make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=1.)
    kind, params = spec["kind"], spec.get("params", {})
    if kind == "svr":
        estimator = SVR(cache_size=256, **params)
    elif kind == "kernelridge":
        estimator = KernelRidge(**params)
    elif kind == "elasticnet":
        estimator = ElasticNet(max_iter=30000, tol=1e-5, **params)
    elif kind == "ridge":
        estimator = Ridge(solver="lsqr", **params)
    elif kind == "huber":
        estimator = HuberRegressor(max_iter=1000, **params)
    else:
        raise ValueError(kind)
    pipeline = Pipeline([
        ("features", CompactPhysicalFeatures(ordinal_only=spec.get("ordinal_only", True), ages_only=spec.get("ages_only", True))),
        ("preprocess", prep), ("model", estimator),
    ])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler(with_std=False))

