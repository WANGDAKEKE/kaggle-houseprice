"""Two bounded exploratory TabPFN v2 variants; no old model refits.

Both variants reuse seed5509's exact full-train folds and cached v5 reference.
Training-only trimming never removes a validation row.  The quality variant is
a categorical representation ablation: string sorting preserves readable level
names, but TabPFN categorical features do not impose ordinal monotonicity.
These exploratory comparisons reuse historical houses and are not independent
validation.  This script never writes the project's main submission_v6.csv.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.house_prices_v6 import LocalTabPFN, CHECKPOINT_SHA256

ART = ROOT / "experiments/v6/variants"
SEED = 5509
N_ESTIMATORS = 4
VARIANTS = ("raw_trim2_e4", "raw_categorical_quality_e4")
SOURCE_FILES = ["scripts/tabpfn_variants_v6.py", "src/house_prices_v6.py", "src/house_prices.py",
                "data/raw/train.csv", "data/raw/test.csv", "requirements-tabpfn.lock.txt",
                "experiments/v6/reference/full1460_seed5509.csv"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def data():
    opts = {"keep_default_na": False, "na_values": ["NA"]}
    train = pd.read_csv(ROOT / "data/raw/train.csv", **opts)
    test = pd.read_csv(ROOT / "data/raw/test.csv", **opts)
    assert len(train) == 1460 and len(test) == 1459 and train.Id.is_unique
    reference = pd.read_csv(ROOT / "experiments/v6/reference/full1460_seed5509.csv")
    assert np.array_equal(reference.Id.to_numpy(), train.Id.to_numpy())
    y = np.log(train.SalePrice.to_numpy())
    assert np.allclose(reference.ylog.to_numpy(), y, rtol=0, atol=1e-13)
    assert np.isfinite(reference.v5_log).all()
    return train, test, reference, y


def feature_frame(frame, variant):
    if variant != "raw_categorical_quality_e4":
        return frame
    transformed = frame.copy()
    for column in ["OverallQual", "OverallCond"]:
        values = transformed[column].to_numpy()
        assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
        transformed[column] = [f"quality_{int(value):02d}" for value in values]
        assert not pd.api.types.is_numeric_dtype(transformed[column].dtype)
    return transformed


def fit_rows_for(train, rows, variant):
    if variant == "raw_trim2_e4":
        subset = train.iloc[rows]
        keep = ~((subset.GrLivArea.to_numpy() > 4000) & (subset.SalePrice.to_numpy() < 300000))
        return rows[keep], rows[~keep]
    return rows, np.array([], dtype=int)


def metrics(y, prediction, frame):
    extreme = (frame.GrLivArea.to_numpy() > 4000) & (frame.SalePrice.to_numpy() < 300000)
    groups = {"all": np.ones(len(y), dtype=bool), "ordinary": ~extreme,
              "normal": frame.SaleCondition.eq("Normal").to_numpy()}
    assert np.isfinite(prediction).all()
    return {name: float(np.sqrt(np.mean((prediction[mask] - y[mask])**2)))
            for name, mask in groups.items()}


def splits(train, reference, variant):
    rows = np.arange(len(train))
    result = []
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(rows)):
        fit, dropped = fit_rows_for(train, tr, variant)
        assert not set(fit).intersection(va)
        assert set(fit).union(dropped) == set(tr)
        assert np.array_equal(reference.fold.to_numpy()[va], np.full(len(va), fold))
        result.append((fold, fit, va, dropped))
    assert sorted(np.concatenate([s[2] for s in result]).tolist()) == rows.tolist()
    return result


def resolved_device(requested):
    if requested != "auto":
        return requested
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_signature(variant, device, fold_plan, train):
    source_hashes = {p: digest(ROOT / p) for p in SOURCE_FILES}
    details = {"variant": variant, "seed": SEED, "n_estimators": N_ESTIMATORS,
        "model_seed_rule": "42+fold", "device": device, "checkpoint": CHECKPOINT_SHA256,
        "source_sha256": source_hashes, "folds": [{"fold": fold,
            "fit_ids": train.Id.iloc[fit].tolist(), "validation_ids": train.Id.iloc[va].tolist(),
            "dropped_training_ids": train.Id.iloc[dropped].tolist()}
            for fold, fit, va, dropped in fold_plan]}
    signature = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return signature, details


def clear_memory(device):
    gc.collect()
    if device == "mps":
        import torch
        torch.mps.empty_cache()


def plan(variant):
    train, _, ref, _ = data()
    fold_plan = splits(train, ref, variant)
    record = {"variant": variant, "seed": SEED, "estimators": N_ESTIMATORS,
        "no_training_performed": True,
        "validation_rows_removed": 0,
        "purpose": "Exploratory variant on reused historical full-train folds; not independent validation",
        "quality_encoding_note": "quality_01..quality_10 are strings and inferred categorical indices; no enforced ordinal monotonicity",
        "folds": [{"fold": fold, "fit_rows": len(fit), "validation_rows": len(va),
                   "dropped_training_ids": train.Id.iloc[dropped].tolist()}
                  for fold, fit, va, dropped in fold_plan]}
    json_write(ART / f"{variant}_plan.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def cv(variant, requested_device):
    train, _, ref, y = data()
    fold_plan = splits(train, ref, variant)
    device = resolved_device(requested_device)
    signature, details = make_signature(variant, device, fold_plan, train)
    X = feature_frame(train, variant)
    path = ART / f"{variant}_seed{SEED}.npz"
    prediction = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1)
    fold_seconds = np.zeros(5)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved["signature"]) == signature, "Variant sources/settings changed; preserve cache and use a new name"
            assert np.array_equal(saved["ids"], train.Id.to_numpy())
            prediction, fold_ids = saved["prediction"].copy(), saved["fold_ids"].copy()
            fold_seconds = saved["fold_seconds"].copy()
            for fold, fit, va, _ in fold_plan:
                if np.isfinite(prediction[va]).any():
                    assert np.isfinite(prediction[va]).all() and (fold_ids[va] == fold).all()
                    assert np.array_equal(saved[f"fit_ids_fold{fold}"], train.Id.iloc[fit].to_numpy())
    records = []
    for fold, fit, va, dropped in fold_plan:
        if not np.isfinite(prediction[va]).all():
            started = time.time()
            model = LocalTabPFN(N_ESTIMATORS, "raw", device, 42+fold).fit(X.iloc[fit], y[fit])
            if variant == "raw_categorical_quality_e4":
                for column in ["OverallQual", "OverallCond"]:
                    assert model.columns_.index(column) in model.categorical_indices_
            prediction[va] = model.predict(X.iloc[va])
            fold_ids[va] = fold
            fold_seconds[fold] = time.time() - started
            extra = {f"fit_ids_fold{f}": train.Id.iloc[fit_rows].to_numpy()
                     for f, fit_rows, _, _ in fold_plan}
            np.savez_compressed(path, prediction=prediction, fold_ids=fold_ids, ids=train.Id.to_numpy(),
                                signature=signature, fold_seconds=fold_seconds, **extra)
            del model
            clear_memory(device)
        baseline = ref.v5_log.to_numpy()[va]
        row = {"variant": variant, "fold": fold, "training_rows": len(fit),
               "validation_rows": len(va), "dropped_training_ids": train.Id.iloc[dropped].tolist(),
               "seconds": float(fold_seconds[fold]), "tabpfn": metrics(y[va], prediction[va], train.iloc[va]),
               "v5": metrics(y[va], baseline, train.iloc[va]),
               "half_log_blend": metrics(y[va], .5*(baseline+prediction[va]), train.iloc[va])}
        records.append(row)
        json_write(ART / f"{variant}_fold_results.json", records)
        print(json.dumps(row), flush=True)
    assert np.isfinite(prediction).all() and np.array_equal(fold_ids, ref.fold.to_numpy())
    comparison = []
    for weight in [0., .25, .5, .75, 1.]:
        blend = (1.-weight)*ref.v5_log.to_numpy() + weight*prediction
        comparison.append({"variant": variant, "tabpfn_weight": weight,
                           **metrics(y, blend, train)})
    table = pd.DataFrame(comparison)
    table.to_csv(ART / f"{variant}_comparison.csv", index=False)
    pd.DataFrame({"Id": train.Id, "ylog": y, "fold": fold_ids, "v5_log": ref.v5_log,
                  "variant_log": prediction}).to_csv(ART / f"{variant}_oof.csv", index=False)
    json_write(ART / f"{variant}_audit.json", {"signature": signature, **details,
        "cache_sha256": digest(path), "all_validation_rows_preserved": True,
        "evaluation_role": "Exploratory model representation/training-context ablation on reused historical folds; not independent validation",
        "scores": comparison, "total_fit_predict_seconds": float(fold_seconds.sum())})
    print(table.to_string(index=False), flush=True)


def test_direction(variant, requested_device):
    train, test, _, y = data()
    audit_path = ART / f"{variant}_audit.json"
    assert audit_path.exists(), "Run five-fold cv first and inspect results before making test direction"
    audit = json.loads(audit_path.read_text())
    device = resolved_device(requested_device)
    for source, expected in audit["source_sha256"].items():
        assert digest(ROOT / source) == expected, source
    fit, dropped = fit_rows_for(train, np.arange(len(train)), variant)
    train_x, test_x = feature_frame(train, variant), feature_frame(test, variant)
    start = time.time()
    model = LocalTabPFN(N_ESTIMATORS, "raw", device, 42).fit(train_x.iloc[fit], y[fit])
    prediction = model.predict(test_x)
    del model
    clear_memory(device)
    baseline = pd.read_csv(ROOT / "submission_v5.csv")
    assert np.array_equal(baseline.Id, test.Id)
    ref = np.log(baseline.SalePrice.to_numpy())
    output = ART / f"{variant}_test_predictions.csv"
    pd.DataFrame({"Id": test.Id, "variant_log": prediction, "v5_log": ref}).to_csv(output, index=False)
    for weight in [.25, .5, .75]:
        candidate = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp((1.-weight)*ref+weight*prediction)})
        assert np.isfinite(candidate.SalePrice).all() and (candidate.SalePrice > 0).all()
        candidate.to_csv(ART / f"candidate_{variant}_w{weight:.2f}.csv", index=False)
    json_write(ART / f"{variant}_test_audit.json", {"variant": variant, "device": device,
        "fit_rows": len(fit), "dropped_training_ids": train.Id.iloc[dropped].tolist(),
        "seconds": time.time()-start, "test_predictions_sha256": digest(output),
        "baseline_submission_sha256": digest(ROOT / "submission_v5.csv"),
        "cv_audit_sha256": digest(audit_path), "source_sha256": audit["source_sha256"],
        "main_submission_v6_written": False, "submitted_to_kaggle": False})
    print("Independent test direction written:", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["plan", "cv", "test"])
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    chosen = VARIANTS if args.variant == "all" else [args.variant]
    for variant in chosen:
        if args.stage == "plan":
            plan(variant)
        elif args.stage == "cv":
            cv(variant, args.device)
        else:
            test_direction(variant, args.device)


if __name__ == "__main__":
    main()
