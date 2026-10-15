"""Bounded kernel/linear pilot on the reused original development split.

This is screening, not fresh confirmation. Every validation row remains in
primary metrics. Excluding two extreme houses is a labelled diagnostic only.
"""
from pathlib import Path
import os
import sys
import time
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from src.house_prices_v5_kernel import build_model
from scripts.run_experiments import load_data
from scripts.optimize_v2 import partitions

ART = ROOT / "experiments/v5/kernel_screen"


def specifications():
    specs = {}
    def add(name, kind, params, **kw):
        specs[name] = {"kind": kind, "params": params, **kw}
    add("svr_compact_g0001_C30", "svr", {"C": 30., "gamma": .0001, "epsilon": .01})
    add("svr_compact_g0003_C10", "svr", {"C": 10., "gamma": .0003, "epsilon": .01})
    add("svr_compact_g001_C5", "svr", {"C": 5., "gamma": .001, "epsilon": .01})
    add("svr_compact_robust", "svr", {"C": 20., "gamma": .0003, "epsilon": .01}, scale="robust")
    add("svr_compact_scaledcats", "svr", {"C": 10., "gamma": .0003, "epsilon": .01}, category_scale=.25)
    add("krr_compact_rbf", "kernelridge", {"alpha": .005, "kernel": "rbf", "gamma": .0005})
    add("krr_compact_laplacian", "kernelridge", {"alpha": .01, "kernel": "laplacian", "gamma": .003})
    add("krr_compact_poly", "kernelridge", {"alpha": .05, "kernel": "polynomial", "degree": 2, "gamma": .001, "coef0": 1.})
    add("en_compact", "elasticnet", {"alpha": .0006, "l1_ratio": .8})
    add("en_compact_scaledcats", "elasticnet", {"alpha": .0007, "l1_ratio": .8}, category_scale=.25)
    add("en_compact_robust", "elasticnet", {"alpha": .0007, "l1_ratio": .8}, scale="robust")
    add("ridge_compact_scaledcats", "ridge", {"alpha": 20.}, category_scale=.25)
    return specs


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p)**2)))


def main():
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "cv").mkdir(exist_ok=True)
    specs = specifications()
    (ART / "specifications.json").write_text(json.dumps(specs, indent=2) + "\n")
    train, _ = load_data()
    dev, _ = partitions(train)
    X = train.iloc[dev]
    y = np.log(X.SalePrice.to_numpy())
    ids = X.Id.to_numpy()
    masks = {"all": np.ones(len(dev), dtype=bool), "without_extremes_diagnostic": ~np.isin(ids, [524, 1299]),
             "normal_sales": X.SaleCondition.eq("Normal").to_numpy()}
    names = ["ref_v2_linear", "ref_v2_cat", "en_inter_size_scaled", "cat_d5_onehot"]
    reference = np.mean([np.load(ROOT / f"experiments/v3/cv/dev_{n}_seed42.npz")["prediction"] for n in names], axis=0)
    folds = np.load(ROOT / "experiments/v3/cv/dev_ref_v2_linear_seed42.npz")["fold_ids"]
    all_predictions = {"Id": ids, "log_target": y, "v4_reference": reference, "fold": folds}
    results = []
    for name, spec in specs.items():
        digest = hashlib.sha256()
        for source in ["src/house_prices.py", "src/house_prices_v2.py", "src/house_prices_v3.py",
                       "src/house_prices_v5_kernel.py", "data/raw/train.csv", "requirements.lock.txt"]:
            digest.update((ROOT / source).read_bytes())
        digest.update(json.dumps({"spec": spec, "seed": 42, "dev_indices": dev.tolist(), "folds": 5}, sort_keys=True).encode())
        signature = digest.hexdigest()
        cache = ART / "cv" / f"{name}__seed42.npz"
        started = time.time()
        if cache.exists() and str(np.load(cache)["signature"]) == signature:
            saved = np.load(cache)
            p, seconds = saved["prediction"], float(saved["seconds"])
        else:
            p = np.full(len(dev), np.nan)
            for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(dev)):
                assert np.all(folds[va] == fold)
                with threadpool_limits(limits=2):
                    model = build_model(spec, seed=42+fold)
                    model.fit(X.iloc[tr], y[tr])
                    p[va] = model.predict(X.iloc[va])
            seconds = time.time()-started
            assert np.isfinite(p).all()
            np.savez_compressed(cache, prediction=p, ids=ids, fold_ids=folds, seconds=seconds, signature=signature)
        all_predictions[name] = p
        row = {"model": name, "seconds": seconds}
        for label, mask in masks.items():
            row[label+"_rmse"] = rmse(y[mask], p[mask])
            row[label+"_v4_reference_rmse"] = rmse(y[mask], reference[mask])
            row[label+"_quarter_graft_rmse"] = rmse(y[mask], (.75*reference+.25*p)[mask])
            row[label+"_half_graft_rmse"] = rmse(y[mask], (.5*reference+.5*p)[mask])
        row["ordinary_positive_mse_gain_fraction"] = float(np.mean(((y-p)**2 < (y-reference)**2)[masks["without_extremes_diagnostic"]]))
        results.append(row)
        pd.DataFrame(results).sort_values("without_extremes_diagnostic_quarter_graft_rmse").to_csv(ART / "screen_results.csv", index=False)
        pd.DataFrame(all_predictions).to_csv(ART / "development_oof.csv", index=False)
        print(json.dumps(row), flush=True)
    manifest = {"seed": 42, "folds": 5, "development_rows": len(dev), "primary_validation_rows_removed": 0,
                "role": "screening on reused development houses; not fresh validation", "threads": 2,
                "reference": "fixed equal log-space blend of frozen v2/v3 four components", "source_sha256": hashlib.sha256((ROOT / "src/house_prices_v5_kernel.py").read_bytes()).hexdigest()}
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
