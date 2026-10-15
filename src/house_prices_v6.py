"""TabPFN v2 local regression. Built with PriorLabs-TabPFN.

Pretrained weights and their license are stored in experiments/v6/pretrained/.
Data-specific transforms fit only on the current training subset.
"""
from pathlib import Path
import os,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TABPFN_NO_BROWSER","true")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR",str(ROOT/".cache/tabpfn/models"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("DO_NOT_TRACK","1")
os.environ.setdefault("TABPFN_MPS_MEMORY_FRACTION","0.5")
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from src.house_prices import Features


CHECKPOINT=ROOT/"experiments/v6/pretrained/tabpfn-v2-regressor.ckpt"
CHECKPOINT_SHA256="2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


class LocalTabPFN(RegressorMixin,BaseEstimator):
    def __init__(self,n_estimators=4,features="raw",device="mps",seed=42):
        self.n_estimators=n_estimators;self.features=features;self.device=device;self.seed=seed

    def frame(self,X):
        if self.features=="physical":
            frame=self.feature_transformer_.transform(X)
        else:frame=X.drop(columns=["Id","SalePrice"],errors="ignore").copy()
        frame["MSSubClass"]=frame.MSSubClass.astype(str)
        for c in frame:
            if not pd.api.types.is_numeric_dtype(frame[c].dtype):
                frame[c]=frame[c].astype(object).where(frame[c].notna(),np.nan)
        return frame

    def fit(self,X,y):
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==CHECKPOINT_SHA256
        if self.features=="physical":self.feature_transformer_=Features(engineered=True,log_numeric=False).fit(X)
        frame=self.frame(X)
        self.categorical_indices_=[i for i,c in enumerate(frame.columns) if not pd.api.types.is_numeric_dtype(frame[c].dtype)]
        self.columns_=frame.columns.tolist()
        torch.set_num_threads(4)
        self.model_=TabPFNRegressor.create_default_for_version(ModelVersion.V2,
            model_path=str(CHECKPOINT),device=self.device,n_estimators=self.n_estimators,
            categorical_features_indices=self.categorical_indices_,random_state=self.seed,
            n_preprocessing_jobs=1,inference_precision=torch.float32,fit_mode="fit_preprocessors",
            memory_saving_mode="auto",ignore_pretraining_limits=(self.device=="cpu"))
        self.model_.fit(frame,np.asarray(y))
        return self

    def predict(self,X):
        frame=self.frame(X)
        assert frame.columns.tolist()==self.columns_
        p=np.asarray(self.model_.predict(frame))
        assert np.isfinite(p).all()
        return p
