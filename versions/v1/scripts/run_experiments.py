"""Predefined 5-fold model comparison, one locked holdout check, final training.

All targets use natural log(SalePrice), matching the stated competition metric.
No Kaggle submission or leaderboard lookup is performed by this script.
"""
from pathlib import Path
import sys
import os
import argparse
import json
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from src.house_prices import build_model, log_rmse

ART = ROOT / "artifacts"
RAW = ROOT / "data" / "raw"
SEED = 42


def specs():
    entries = [
        ("ridge_baseline", "ridge", False, False, {"alpha": 20}),
        ("random_forest_baseline", "randomforest", False, False, {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.7}),
        ("ridge_features_a10", "ridge", True, True, {"alpha": 10}),
        ("ridge_features_a30", "ridge", True, True, {"alpha": 30}),
        ("ridge_features_a100", "ridge", True, True, {"alpha": 100}),
        ("lasso_features_a0003", "lasso", True, True, {"alpha": 0.0003}),
        ("lasso_features_a0006", "lasso", True, True, {"alpha": 0.0006}),
        ("elasticnet_features", "elasticnet", True, True, {"alpha": 0.0007, "l1_ratio": 0.5}),
        ("gbr_huber", "gbr", True, True, {"n_estimators": 1800, "learning_rate": 0.025, "max_depth": 3, "min_samples_leaf": 10, "max_features": 0.8, "loss": "huber"}),
        ("catboost_raw_d5", "catboost", False, False, {"iterations": 2200, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 5}),
        ("catboost_features_d4", "catboost", True, False, {"iterations": 2500, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5}),
        ("catboost_features_d5", "catboost", True, False, {"iterations": 2200, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 5}),
        ("xgboost_features", "xgboost", True, True, {"n_estimators": 1800, "learning_rate": 0.025, "max_depth": 3, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5, "reg_alpha": 0.02}),
        ("lightgbm_features", "lightgbm", True, True, {"n_estimators": 1400, "learning_rate": 0.025, "num_leaves": 12, "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 3, "reg_alpha": 0.02}),
    ]
    return {name: {"kind": kind, "engineered": engineered, "log_numeric": log_numeric,
                   "params": params, "schema_path": str(RAW / "train.csv")}
            for name, kind, engineered, log_numeric, params in entries}


def load_data():
    train = pd.read_csv(RAW / "train.csv", keep_default_na=False, na_values=["NA"])
    test = pd.read_csv(RAW / "test.csv", keep_default_na=False, na_values=["NA"])
    sample = pd.read_csv(RAW / "sample_submission.csv")
    assert train.shape == (1460, 81) and test.shape == (1459, 80)
    assert train.Id.is_unique and test.Id.is_unique
    assert list(train.drop(columns="SalePrice").columns) == list(test.columns)
    assert np.array_equal(test.Id, sample.Id)
    assert set(train.Id).isdisjoint(set(test.Id)) and (train.SalePrice > 0).all()
    return train, test


def compare():
    train, _ = load_data()
    y = np.log(train.SalePrice.to_numpy())
    dev, hold = train_test_split(np.arange(len(train)), test_size=0.2, random_state=SEED,
                                 stratify=pd.qcut(train.SalePrice, 10, labels=False))
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(dev))
    split_frame = pd.DataFrame({"Id": train.Id, "partition": "holdout", "cv_fold": -1})
    split_frame.loc[dev, "partition"] = "development"
    for k, (_, va) in enumerate(folds):
        split_frame.loc[dev[va], "cv_fold"] = k
    split_frame.to_csv(ART / "splits.csv", index=False)
    config = specs()
    (ART / "experiment_specs.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, oof = [], pd.DataFrame({"Id": train.Id.iloc[dev].to_numpy(), "log_target": y[dev]})
    # Persist each experiment so a long run can resume without repeating fits.
    for name, spec in config.items():
        cache = ART / "cv" / f"{name}.npz"
        start = time.time()
        if cache.exists():
            values = np.load(cache)
            prediction, fold_scores = values["prediction"], values["fold_scores"].tolist()
            duration = float(values["seconds"])
            print(f"Cached {name}", flush=True)
        else:
            prediction = np.full(len(dev), np.nan)
            fold_scores = []
            try:
                for k, (tr, va) in enumerate(folds):
                    model = build_model(spec, SEED + k)
                    model.fit(train.iloc[dev[tr]], y[dev[tr]])
                    prediction[va] = model.predict(train.iloc[dev[va]])
                    fold_scores.append(log_rmse(y[dev[va]], prediction[va]))
                    print(f"{name} fold {k + 1}/5: {fold_scores[-1]:.5f}", flush=True)
                duration = time.time() - start
                np.savez(cache, prediction=prediction, fold_scores=fold_scores, seconds=duration)
            except (ImportError, OSError) as exc:
                print(f"UNAVAILABLE {name}: {exc}", flush=True)
                (ART / "cv" / f"{name}_unavailable.txt").write_text(str(exc))
                continue
        assert np.isfinite(prediction).all()
        oof[name] = prediction
        rows.append({"model": name, "oof_log_rmse": log_rmse(y[dev], prediction),
                     "fold_mean": np.mean(fold_scores), "fold_std": np.std(fold_scores),
                     "seconds": duration, **{f"fold_{k + 1}": v for k, v in enumerate(fold_scores)}})
        pd.DataFrame(rows).sort_values("oof_log_rmse").to_csv(ART / "experiments.csv", index=False)
        oof.to_csv(ART / "development_oof.csv", index=False)
    # Select one representative from each family using development OOF only.
    scores = {row["model"]: row["oof_log_rmse"] for row in rows}
    linear = min([n for n in scores if config[n]["kind"] in ["ridge", "lasso", "elasticnet"]], key=scores.get)
    tree = min([n for n in scores if config[n]["kind"] in ["catboost", "gbr", "xgboost", "lightgbm"]], key=scores.get)
    blends = {
        "blend_linear25_tree75": {linear: 0.25, tree: 0.75},
        "blend_linear50_tree50": {linear: 0.5, tree: 0.5},
        "blend_linear75_tree25": {linear: 0.75, tree: 0.25},
    }
    # A small, fixed candidate set rather than optimizing dozens of blend weights.
    if "gbr_huber" in scores and tree != "gbr_huber":
        blends["blend_three_families"] = {linear: 0.4, tree: 0.4, "gbr_huber": 0.2}
    for name, weights in blends.items():
        prediction = sum(oof[n].to_numpy() * w for n, w in weights.items())
        oof[name] = prediction
        fold_scores = [log_rmse(y[dev[va]], prediction[va]) for _, va in folds]
        rows.append({"model": name, "oof_log_rmse": log_rmse(y[dev], prediction),
                     "fold_mean": np.mean(fold_scores), "fold_std": np.std(fold_scores),
                     "seconds": 0, **{f"fold_{k + 1}": v for k, v in enumerate(fold_scores)}})
    results = pd.DataFrame(rows).sort_values("oof_log_rmse")
    results.to_csv(ART / "experiments.csv", index=False)
    oof.to_csv(ART / "development_oof.csv", index=False)
    # Prefer fewer components when differences are negligible on development CV.
    # This rule is applied before opening the locked holdout.
    best_score = float(results.iloc[0].oof_log_rmse)
    tolerance = 0.0003
    near_best = results.loc[results.oof_log_rmse <= best_score + tolerance].copy()
    near_best["components"] = near_best.model.map(lambda n: len(blends.get(n, {n: 1.0})))
    chosen = near_best.sort_values(["components", "oof_log_rmse"]).iloc[0]
    winner = chosen.model
    weights = blends.get(winner, {winner: 1.0})
    selection = {"name": winner, "weights": weights, "selection_cv_log_rmse": float(chosen.oof_log_rmse),
                 "best_candidate_cv_log_rmse": best_score, "simplicity_tolerance": tolerance,
                 "selection_rule": "within 0.0003 of best development OOF: fewest components, then lowest RMSE",
                 "development_rows": len(dev), "holdout_rows": len(hold), "seed": SEED,
                 "target": "natural log(SalePrice)", "holdout_policy": "one-time check; no retuning on holdout"}
    (ART / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(results[["model", "oof_log_rmse", "fold_std", "seconds"]].to_string(index=False), flush=True)
    print("Selected:", json.dumps(selection), flush=True)


def finalize():
    train, test = load_data()
    y = np.log(train.SalePrice.to_numpy())
    split_frame = pd.read_csv(ART / "splits.csv")
    assert np.array_equal(split_frame.Id, train.Id)
    dev = np.flatnonzero(split_frame.partition == "development")
    hold = np.flatnonzero(split_frame.partition == "holdout")
    selection = json.loads((ART / "selection.json").read_text())
    config = json.loads((ART / "experiment_specs.json").read_text())
    weights = selection["weights"]
    # Evaluate exactly the chosen fixed blend and the baseline on untouched holdout.
    hold_predictions = {}
    for name in sorted(set(weights) | {"ridge_baseline"}):
        model = build_model(config[name], SEED)
        model.fit(train.iloc[dev], y[dev])
        hold_predictions[name] = model.predict(train.iloc[hold])
    chosen_hold = sum(hold_predictions[n] * w for n, w in weights.items())
    hold_frame = train.iloc[hold][["Id", "SalePrice", "GrLivArea", "OverallQual"]].copy()
    hold_frame["log_target"] = y[hold]
    hold_frame["log_prediction"] = chosen_hold
    hold_frame["predicted_price"] = np.exp(chosen_hold)
    hold_frame["log_error"] = chosen_hold - y[hold]
    hold_frame.to_csv(ART / "holdout_predictions.csv", index=False)
    hold_report = {"selected_log_rmse": log_rmse(y[hold], chosen_hold),
                   "baseline_log_rmse": log_rmse(y[hold], hold_predictions["ridge_baseline"]),
                   "n": len(hold), "selected_model": selection["name"],
                   "individual_selected_models": {n: log_rmse(y[hold], hold_predictions[n]) for n in weights}}
    (ART / "holdout_metrics.json").write_text(json.dumps(hold_report, indent=2) + "\n")
    print("Untouched holdout:", json.dumps(hold_report), flush=True)
    # Refit on every official training row after model selection is frozen.
    predicted_log = np.zeros(len(test))
    component_frame = pd.DataFrame({"Id": test.Id})
    for name, weight in weights.items():
        model = build_model(config[name], SEED)
        model.fit(train, y)
        component_log = model.predict(test)
        predicted_log += weight * component_log
        component_frame[name] = component_log
        joblib.dump(model, ART / "models" / f"{name}.joblib")
        print(f"Refit all training rows: {name}", flush=True)
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(predicted_log)})
    validate_submission(submission, test)
    submission.to_csv(ROOT / "submission.csv", index=False)
    component_frame.to_csv(ART / "test_component_log_predictions.csv", index=False)
    # Also keep a simple reproducible reference submission for later comparison.
    baseline = build_model(config["ridge_baseline"], SEED)
    baseline.fit(train, y)
    baseline_submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(baseline.predict(test))})
    validate_submission(baseline_submission, test)
    baseline_submission.to_csv(ROOT / "submission_baseline.csv", index=False)
    summary = {"rows": len(submission), "columns": list(submission.columns),
               "min_price": float(submission.SalePrice.min()), "max_price": float(submission.SalePrice.max()),
               "median_price": float(submission.SalePrice.median()), "finite_positive": True,
               "id_order_matches_official_test": True, "submitted_to_kaggle": False}
    (ART / "submission_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Submission validated:", json.dumps(summary), flush=True)


def validate_submission(submission, test):
    assert list(submission.columns) == ["Id", "SalePrice"]
    assert len(submission) == len(test) and submission.Id.is_unique
    assert np.array_equal(submission.Id, test.Id)
    assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice > 0).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["compare", "finalize", "all"])
    args = parser.parse_args()
    for directory in [ART, ART / "cv", ART / "models"]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage in ["compare", "all"]:
        compare()
    if args.stage in ["finalize", "all"]:
        finalize()
