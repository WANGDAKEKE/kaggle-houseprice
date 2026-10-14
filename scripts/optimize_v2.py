"""v2: fold-local model search, repeat-split confirmation, and versioned predictions.

The original holdout is already used by v1; it is a diagnostic only in v2.
All screening/selection uses the original development partition, not the holdout.
"""
from pathlib import Path
import sys
import os
import argparse
import json
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from src.house_prices import log_rmse
from src.house_prices_v2 import build_v2_model, training_weights
from scripts.run_experiments import load_data, validate_submission

ART = ROOT / "experiments" / "v2"
CONFIRM_SEEDS = [42, 2026, 137]


def specifications():
    v1 = json.loads((ROOT / "versions/v1/artifacts/experiment_specs.json").read_text())
    config = {"v1_elasticnet": {**v1["elasticnet_features"], "features": "v1"},
              "v1_catboost": {**v1["catboost_features_d5"], "features": "v1"}}
    def add(name, kind, params, **kw):
        config[name] = {"kind": kind, "params": params, **kw}
    for alpha, ratio in [(0.0003, 0.5), (0.0007, 0.5), (0.001, 0.3)]:
        add(f"en_domain_a{alpha}_l{ratio}", "elasticnet", {"alpha": alpha, "l1_ratio": ratio})
    add("en_rich", "elasticnet", {"alpha": 0.0007, "l1_ratio": 0.5}, rich=True)
    add("en_cap995", "elasticnet", {"alpha": 0.0007, "l1_ratio": 0.5}, cap_quantile=0.995)
    add("lasso_domain", "lasso", {"alpha": 0.0005})
    add("ridge_domain_a10", "ridge", {"alpha": 10}, min_frequency=3)
    add("ridge_rich_a30", "ridge", {"alpha": 30}, rich=True, min_frequency=3)
    for c, gamma in [(10, .003), (20, .003), (50, .003), (20, .001), (20, .01)]:
        add(f"svr_C{c}_g{gamma}", "svr", {"C": c, "gamma": gamma, "epsilon": .008})
    add("svr_cap995", "svr", {"C": 20, "gamma": .003, "epsilon": .008}, cap_quantile=.995)
    add("svr_rich", "svr", {"C": 20, "gamma": .003, "epsilon": .008}, rich=True)
    add("svr_robust", "svr", {"C": 20, "gamma": .0003, "epsilon": .008}, scale="robust")
    for alpha, coef in [(.1, 1), (1, 1), (.6, 2.5)]:
        add(f"krr_poly_a{alpha}_c{coef}", "kernelridge", {"alpha": alpha, "kernel": "polynomial", "degree": 2, "gamma": .001, "coef0": coef})
    add("krr_rbf", "kernelridge", {"alpha": .03, "kernel": "rbf", "gamma": .003})
    for depth, reg in [(4, 3), (4, 8), (5, 5), (6, 8)]:
        add(f"cat_domain_d{depth}_l{reg}", "catboost", {"iterations": 2600, "learning_rate": .03, "depth": depth, "l2_leaf_reg": reg}, log_numeric=False)
    add("cat_rich_d5", "catboost", {"iterations": 2600, "learning_rate": .03, "depth": 5, "l2_leaf_reg": 5}, log_numeric=False, rich=True)
    add("cat_domain_downweight", "catboost", {"iterations": 2600, "learning_rate": .03, "depth": 5, "l2_leaf_reg": 5}, log_numeric=False, sample_weight="downweight_large")
    add("xgb_domain_d2", "xgboost", {"n_estimators": 2000, "learning_rate": .025, "max_depth": 2, "min_child_weight": 3, "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5, "reg_alpha": .05}, dense=True)
    add("xgb_domain_d3", "xgboost", {"n_estimators": 1800, "learning_rate": .025, "max_depth": 3, "min_child_weight": 3, "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5, "reg_alpha": .05}, dense=True)
    add("lgb_domain_l8", "lightgbm", {"n_estimators": 1800, "learning_rate": .02, "num_leaves": 8, "min_child_samples": 15, "colsample_bytree": .8, "reg_lambda": 3})
    add("lgb_rich_l15", "lightgbm", {"n_estimators": 1600, "learning_rate": .02, "num_leaves": 15, "min_child_samples": 15, "colsample_bytree": .8, "reg_lambda": 5}, rich=True)
    # Follow-up development experiments: refine the improved rich linear family
    # and correct the missing intercept of the initial RBF KernelRidge variants.
    for alpha, ratio in [(.0003, .5), (.001, .5), (.0007, .2), (.0007, .8)]:
        add(f"en_rich_a{alpha}_l{ratio}", "elasticnet", {"alpha": alpha, "l1_ratio": ratio}, rich=True)
    add("lasso_rich_a0003", "lasso", {"alpha": .0003}, rich=True)
    add("lasso_rich_a0007", "lasso", {"alpha": .0007}, rich=True)
    for gamma in [.00005, .0001, .0003]:
        add(f"svr_smallgamma_{gamma}", "svr", {"C": 30, "gamma": gamma, "epsilon": .008})
    for gamma, alpha in [(.0003, .01), (.003, .01), (.0003, .1)]:
        add(f"krr_centered_g{gamma}_a{alpha}", "kernelridge", {"kernel": "rbf", "gamma": gamma, "alpha": alpha}, center_target=True)
    return config


def partitions(train):
    split = pd.read_csv(ROOT / "versions/v1/artifacts/splits.csv")
    assert np.array_equal(split.Id, train.Id)
    # Match original shuffled development order from v1 for an exact seed-42 comparison.
    original_order = pd.read_csv(ROOT / "versions/v1/artifacts/development_oof.csv").Id.to_numpy()
    id_to_index = pd.Series(np.arange(len(train)), index=train.Id)
    dev = id_to_index.loc[original_order].to_numpy()
    hold = np.flatnonzero(split.partition == "holdout")
    assert set(dev) == set(np.flatnonzero(split.partition == "development"))
    assert set(dev).isdisjoint(set(hold))
    return dev, hold


def cache_signature(name, spec, seed, dev):
    digest = hashlib.sha256()
    for path in [ROOT / "src/house_prices.py", ROOT / "src/house_prices_v2.py", ROOT / "data/raw/train.csv", ROOT / "requirements.lock.txt"]:
        digest.update(path.read_bytes())
    digest.update(json.dumps({"name": name, "spec": spec, "seed": seed, "dev": dev.tolist(), "folds": 5}, sort_keys=True).encode())
    return digest.hexdigest()


def fit_model(spec, train, y, seed):
    model = build_v2_model(spec, seed)
    if spec.get("center_target", False):
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler(with_std=False))
    weights = training_weights(train, y, spec.get("sample_weight", "none"))
    if weights is None:
        model.fit(train, y)
    else:
        model.fit(train, y, model__sample_weight=weights)
    return model


def run_cv(name, spec, seed, train, y, dev):
    path = ART / "cv" / f"{name}__seed{seed}.npz"
    signature = cache_signature(name, spec, seed, dev)
    if path.exists():
        saved = np.load(path)
        if str(saved["signature"]) == signature:
            return saved["prediction"], saved["fold_scores"], float(saved["seconds"])
    start = time.time()
    predictions = np.full(len(dev), np.nan)
    scores = []
    fold_ids = np.full(len(dev), -1)
    for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(dev)):
        model = fit_model(spec, train.iloc[dev[tr]], y[dev[tr]], seed + fold)
        predictions[va] = model.predict(train.iloc[dev[va]])
        fold_ids[va] = fold
        scores.append(log_rmse(y[dev[va]], predictions[va]))
        print(f"{name} seed={seed} fold={fold + 1}: {scores[-1]:.5f}", flush=True)
    assert np.isfinite(predictions).all() and (fold_ids >= 0).all()
    seconds = time.time() - start
    np.savez_compressed(path, prediction=predictions, fold_scores=scores, seconds=seconds, signature=signature, fold_ids=fold_ids, dev_ids=train.Id.iloc[dev].to_numpy())
    return predictions, np.array(scores), seconds


def metrics(name, p, train, y, dev, fold_scores=None, seconds=0):
    anomalous = (train.GrLivArea.iloc[dev].to_numpy() > 4000) & (y[dev] < np.log(300000))
    return {"model": name, "oof_log_rmse": log_rmse(y[dev], p),
            "ordinary_rows_diagnostic_log_rmse": log_rmse(y[dev][~anomalous], p[~anomalous]),
            "fold_std": float(np.std(fold_scores)) if fold_scores is not None else None,
            "seconds": seconds}


def screen():
    train, _ = load_data()
    y = np.log(train.SalePrice.to_numpy())
    dev, _ = partitions(train)
    config = specifications()
    (ART / "specifications.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, predictions = [], {}
    for name, spec in config.items():
        p, scores, seconds = run_cv(name, spec, 42, train, y, dev)
        predictions[name] = p
        rows.append(metrics(name, p, train, y, dev, scores, seconds))
        pd.DataFrame(rows).sort_values("oof_log_rmse").to_csv(ART / "screen_results.csv", index=False)
    baseline = .25 * predictions["v1_elasticnet"] + .75 * predictions["v1_catboost"]
    rows.append(metrics("v1_fixed_blend", baseline, train, y, dev))
    expected = json.loads((ROOT / "versions/v1/artifacts/selection.json").read_text())["selection_cv_log_rmse"]
    assert abs(log_rmse(y[dev], baseline) - expected) < 1e-10, "v1 reference CV failed to reproduce"
    result = pd.DataFrame(rows).sort_values("oof_log_rmse")
    result.to_csv(ART / "screen_results.csv", index=False)
    # Only development screening defines the shortlist; holdout stays out of selection.
    shortlisted = list(result.loc[result.model != "v1_fixed_blend", "model"].head(5))
    for family in [["svr", "kernelridge"], ["ridge", "lasso", "elasticnet"], ["catboost"], ["xgboost", "lightgbm"]]:
        candidates = result.loc[result.model.map(lambda n: n in config and config[n]["kind"] in family)]
        best = candidates.iloc[0].model
        if best not in shortlisted:
            shortlisted.append(best)
    shortlisted = list(dict.fromkeys(shortlisted + ["v1_elasticnet", "v1_catboost"]))
    shortlist = {"models": shortlisted, "confirmation_seeds": CONFIRM_SEEDS,
                 "screen_seed": 42, "holdout_used_for_selection": False}
    (ART / "shortlist.json").write_text(json.dumps(shortlist, indent=2) + "\n")
    print("Screen complete:", result.to_string(index=False), flush=True)
    print("Confirmation shortlist:", shortlisted, flush=True)


def confirm():
    train, _ = load_data()
    y = np.log(train.SalePrice.to_numpy())
    dev, _ = partitions(train)
    config = json.loads((ART / "specifications.json").read_text())
    shortlist = json.loads((ART / "shortlist.json").read_text())["models"]
    predictions, scores, rows = {}, {}, []
    for name in shortlist:
        values = []
        for seed in CONFIRM_SEEDS:
            p, _, _ = run_cv(name, config[name], seed, train, y, dev)
            values.append(p)
        predictions[name] = np.column_stack(values)
        scores[name] = [log_rmse(y[dev], predictions[name][:, i]) for i in range(len(CONFIRM_SEEDS))]
        rows.append({"model": name, "mean_seed_rmse": np.mean(scores[name]),
                     "seed_std": np.std(scores[name]), **{f"seed_{s}": scores[name][i] for i, s in enumerate(CONFIRM_SEEDS)}})
        pd.DataFrame(rows).sort_values("mean_seed_rmse").to_csv(ART / "confirmation_base_results.csv", index=False)
    # Family representatives are chosen on repeated development CV only.
    mean_scores = {n: np.mean(v) for n, v in scores.items()}
    linear = min([n for n in shortlist if config[n]["kind"] in ["ridge", "lasso", "elasticnet"]], key=mean_scores.get)
    kernel = min([n for n in shortlist if config[n]["kind"] in ["svr", "kernelridge"]], key=mean_scores.get)
    tree = min([n for n in shortlist if config[n]["kind"] in ["catboost", "xgboost", "lightgbm"]], key=mean_scores.get)
    candidates = {n: {n: 1.0} for n in shortlist}
    candidates["v1_fixed_blend"] = {"v1_elasticnet": .25, "v1_catboost": .75}
    for weight in [.25, .5, .75]:
        candidates[f"linear_tree_{weight}"] = {linear: weight, tree: 1 - weight}
        candidates[f"kernel_tree_{weight}"] = {kernel: weight, tree: 1 - weight}
    for name, values in [ ("three_equal", [1/3, 1/3, 1/3]), ("three_tree50", [.25, .25, .5]), ("three_tree60", [.2, .2, .6]) ]:
        candidates[name] = dict(zip([linear, kernel, tree], values))
    if linear != "v1_elasticnet":
        candidates["two_linear_tree50"] = {"v1_elasticnet": .25, linear: .25, tree: .5}
        candidates["two_linear_tree60"] = {"v1_elasticnet": .2, linear: .2, tree: .6}
        candidates["two_linear_tree75"] = {"v1_elasticnet": .125, linear: .125, tree: .75}
    results = []
    for name, weights in candidates.items():
        p = sum(predictions[n] * w for n, w in weights.items())
        seed_scores = [log_rmse(y[dev], p[:, i]) for i in range(len(CONFIRM_SEEDS))]
        results.append({"model": name, "mean_seed_rmse": np.mean(seed_scores), "seed_std": np.std(seed_scores),
                        "components": len(weights), **{f"seed_{s}": seed_scores[i] for i, s in enumerate(CONFIRM_SEEDS)}})
    result = pd.DataFrame(results).sort_values("mean_seed_rmse")
    result.to_csv(ART / "confirmation_results.csv", index=False)
    best = result.iloc[0].mean_seed_rmse
    close = result.loc[result.mean_seed_rmse <= best + .0002].sort_values(["components", "mean_seed_rmse"])
    selected = close.iloc[0]
    reference = result.loc[result.model == "v1_fixed_blend"].iloc[0]
    weights = candidates[selected.model]
    selected_predictions = sum(predictions[n] * w for n, w in weights.items())
    reference_predictions = .25 * predictions["v1_elasticnet"] + .75 * predictions["v1_catboost"]
    details = pd.DataFrame({"Id": train.Id.iloc[dev].to_numpy(), "log_target": y[dev]})
    for i, seed in enumerate(CONFIRM_SEEDS):
        details[f"selected_seed_{seed}"] = selected_predictions[:, i]
        details[f"v1_seed_{seed}"] = reference_predictions[:, i]
    details.to_csv(ART / "confirmed_oof.csv", index=False)
    # Paired bootstrap of houses (not independent folds) for development uncertainty.
    rng = np.random.default_rng(20261015)
    selected_sq = np.mean((selected_predictions - y[dev, None]) ** 2, axis=1)
    reference_sq = np.mean((reference_predictions - y[dev, None]) ** 2, axis=1)
    improvement = []
    for _ in range(2000):
        idx = rng.integers(0, len(dev), len(dev))
        improvement.append(np.sqrt(reference_sq[idx].mean()) - np.sqrt(selected_sq[idx].mean()))
    selection = {"version": "v2", "name": selected.model, "weights": weights,
                 "mean_repeated_cv_log_rmse": float(selected.mean_seed_rmse),
                 "v1_mean_repeated_cv_log_rmse": float(reference.mean_seed_rmse),
                 "seeds": CONFIRM_SEEDS,
                 "selected_seed_scores": {str(s): float(selected[f"seed_{s}"]) for s in CONFIRM_SEEDS},
                 "v1_seed_scores": {str(s): float(reference[f"seed_{s}"]) for s in CONFIRM_SEEDS},
                 "paired_bootstrap_improvement_95pct": np.percentile(improvement, [2.5, 97.5]).tolist(),
                 "bootstrap_note": "Exploratory uncertainty after model selection; not an independent confidence guarantee.",
                 "selection_rule": "3-seed 5-fold development CV; within 0.0002 choose fewer components then RMSE",
                 "holdout_used_for_selection": False, "validation_rows_removed": 0,
                 "v1_public_score_user_reported": .11969, "v2_public_score": None,
                 "final_training_policy": "all official rows, average three model seeds for stochastic models"}
    (ART / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)
    print("Frozen v2 selection:", json.dumps(selection), flush=True)


def finalize():
    train, test = load_data()
    y = np.log(train.SalePrice.to_numpy())
    dev, hold = partitions(train)
    config = json.loads((ART / "specifications.json").read_text())
    selection = json.loads((ART / "selection.json").read_text())
    weights = selection["weights"]
    model_entries = []
    prediction = np.zeros(len(test))
    hold_prediction = np.zeros(len(hold))
    components = pd.DataFrame({"Id": test.Id})
    for name, weight in weights.items():
        kind = config[name]["kind"]
        seeds = CONFIRM_SEEDS if kind in ["catboost", "xgboost", "lightgbm"] else [42]
        for seed in seeds:
            model = fit_model(config[name], train.iloc[dev], y[dev], seed)
            hold_prediction += weight / len(seeds) * model.predict(train.iloc[hold])
            model = fit_model(config[name], train, y, seed)
            values = model.predict(test)
            prediction += weight / len(seeds) * values
            model_path = ART / "models" / f"{name}_seed{seed}.joblib"
            joblib.dump(model, model_path)
            components[f"{name}_seed{seed}"] = values
            model_entries.append({"name": name, "seed": seed, "weight": weight / len(seeds), "path": str(model_path.relative_to(ROOT))})
            print(f"Final fit: {name}, seed={seed}", flush=True)
    # Already-used v1 holdout: supporting diagnosis only, never used to choose or retune.
    hold_frame = pd.DataFrame({"Id": train.Id.iloc[hold].to_numpy(), "log_target": y[hold], "log_prediction": hold_prediction})
    hold_frame.to_csv(ART / "holdout_diagnostic_predictions.csv", index=False)
    hold_result = {"log_rmse": log_rmse(y[hold], hold_prediction), "n": len(hold),
                   "role": "diagnostic only; this holdout was already evaluated in v1", "used_for_selection": False}
    (ART / "holdout_diagnostic.json").write_text(json.dumps(hold_result, indent=2) + "\n")
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(prediction)})
    validate_submission(submission, test)
    submission.to_csv(ROOT / "submission_v2.csv", index=False)
    components.to_csv(ART / "test_component_log_predictions.csv", index=False)
    (ART / "model_manifest.json").write_text(json.dumps(model_entries, indent=2) + "\n")
    old = pd.read_csv(ROOT / "versions/v1/submission.csv")
    assert np.array_equal(old.Id, submission.Id)
    comparison = pd.DataFrame({"Id": test.Id, "v1": old.SalePrice, "v2": submission.SalePrice})
    comparison["ratio_v2_v1"] = comparison.v2 / comparison.v1
    comparison.to_csv(ART / "submission_comparison.csv", index=False)
    audit = {"rows": len(submission), "columns": list(submission.columns), "positive_finite": True,
             "test_ids_match": True, "submission_sha256": hashlib.sha256((ROOT / "submission_v2.csv").read_bytes()).hexdigest(),
             "v1_preserved_sha256": hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest(),
             "min_price": float(submission.SalePrice.min()), "max_price": float(submission.SalePrice.max()),
             "median_prediction_ratio_to_v1": float(comparison.ratio_v2_v1.median()),
             "p01_prediction_ratio_to_v1": float(comparison.ratio_v2_v1.quantile(.01)),
             "p99_prediction_ratio_to_v1": float(comparison.ratio_v2_v1.quantile(.99)),
             "submitted_to_kaggle": False}
    (ART / "submission_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("v2 submission:", json.dumps(audit), flush=True)
    print("Old holdout diagnostic:", json.dumps(hold_result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "confirm", "finalize", "all"])
    args = parser.parse_args()
    for d in [ART, ART / "cv", ART / "models"]:
        d.mkdir(parents=True, exist_ok=True)
    if args.stage in ["screen", "all"]:
        screen()
    if args.stage in ["confirm", "all"]:
        confirm()
    if args.stage in ["finalize", "all"]:
        finalize()
