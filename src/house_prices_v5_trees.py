"""Fold-local tree candidates for v5; robust losses start near fold target mean."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from src.house_prices import CatFrame, Features
from src.house_prices_v2 import DomainFeatures, v2_preprocessing, build_v2_model


class CenteredTree(RegressorMixin, BaseEstimator):
    """Fit on centered fold targets, avoiding zero initialization at log(price)=12."""
    def __init__(self, kind="catboost", params=None, seed=42):
        self.kind = kind
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        self.offset_ = float(np.mean(y))
        params = dict(self.params or {})
        if self.kind == "catboost":
            from catboost import CatBoostRegressor
            self.model_ = CatBoostRegressor(random_seed=self.seed, thread_count=2,
                verbose=False, allow_writing_files=False,
                cat_features=list(X.select_dtypes(exclude="number").columns), **params)
        elif self.kind == "lightgbm":
            from lightgbm import LGBMRegressor
            self.model_ = LGBMRegressor(random_state=self.seed, n_jobs=2,
                verbosity=-1, **params)
        elif self.kind == "xgboost":
            from xgboost import XGBRegressor
            self.model_ = XGBRegressor(random_state=self.seed, n_jobs=2,
                tree_method="hist", base_score=0., **params)
        elif self.kind == "gbr":
            self.model_ = GradientBoostingRegressor(random_state=self.seed, **params)
        else:
            raise ValueError(self.kind)
        self.model_.fit(X, np.asarray(y) - self.offset_)
        return self

    def predict(self, X):
        return self.model_.predict(X) + self.offset_


class CrossFittedENResidualCat(RegressorMixin, BaseEstimator):
    """A shallow tree learns only cross-fitted, three-fold linear residuals."""
    def __init__(self, params=None, seed=42):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        target = np.asarray(y)
        linear_spec = {"kind": "elasticnet", "rich": True,
            "params": {"alpha": .0007, "l1_ratio": .8}}
        out_of_fold = np.full(len(X), np.nan)
        for fold, (tr, va) in enumerate(KFold(3, shuffle=True,
                random_state=self.seed + 731).split(X)):
            linear = build_v2_model(linear_spec, self.seed + fold)
            linear.fit(X.iloc[tr], target[tr])
            out_of_fold[va] = linear.predict(X.iloc[va])
        assert np.isfinite(out_of_fold).all()
        self.linear_ = build_v2_model(linear_spec, self.seed)
        self.linear_.fit(X, target)
        self.residual_ = build_model({"kind": "catboost", "features": "v1",
            "params": self.params}, self.seed + 173)
        self.residual_.fit(X, target - out_of_fold)
        return self

    def predict(self, X):
        return self.linear_.predict(X) + self.residual_.predict(X)


def build_model(spec, seed=42):
    """Build a model without learned transforms outside its own training fold."""
    if spec["kind"] == "crossfit_en_residual_cat":
        return CrossFittedENResidualCat(spec.get("params"), seed)
    features = (Features(engineered=True, log_numeric=False) if spec.get("features") == "v1" else
        DomainFeatures(rich=False, log_numeric=spec.get("log_numeric", False)))
    frame = CatFrame() if spec["kind"] == "catboost" else v2_preprocessing(
        scale=None, min_frequency=spec.get("min_frequency", 3), dense=False)
    return Pipeline([("features", features), ("frame", frame),
        ("model", CenteredTree(spec["kind"], spec.get("params"), seed))])
