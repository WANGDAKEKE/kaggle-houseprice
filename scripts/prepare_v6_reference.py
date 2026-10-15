"""Reconstruct fixed v4/v5 OOF references using existing caches only.

This script performs no fitting and does not rerun v5's inner recipe selector.
The final, already selected v5 recipe is applied retrospectively to each cached
outer validation fold.  These reused houses and post-selection comparisons are
not a new independent evaluation of the frozen v5 recipe.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ART = ROOT / "experiments/v6/reference"
BASE_NAMES = ["ref_v2_linear", "ref_v2_cat", "en_inter_size_scaled", "cat_d5_onehot"]
SVR_NAME = "svr_compact_g001_C5"
V5_SOURCES = ["scripts/optimize_v5.py", "src/house_prices.py", "src/house_prices_v2.py",
              "src/house_prices_v3.py", "src/house_prices_v5_kernel.py",
              "src/house_prices_v5_trees.py", "src/house_prices_v5_encoding.py",
              "requirements.lock.txt", "data/raw/train.csv"]
V3_SOURCES = ["src/house_prices.py", "src/house_prices_v2.py", "src/house_prices_v3.py",
              "scripts/optimize_v3.py", "requirements.lock.txt", "data/raw/train.csv"]
KERNEL_SOURCES = ["src/house_prices.py", "src/house_prices_v2.py", "src/house_prices_v3.py",
                  "src/house_prices_v5_kernel.py", "data/raw/train.csv", "requirements.lock.txt"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(paths):
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in paths)).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def support_weights(fit_frame, pred_frame):
    q95, q99 = np.quantile(fit_frame.GrLivArea.to_numpy(), [.95, .99])
    area = pred_frame.GrLivArea.to_numpy()
    if q99 <= q95:
        return (area <= q95).astype(float), [float(q95), float(q99)]
    return np.clip((q99 - area) / (q99 - q95), 0., 1.), [float(q95), float(q99)]


def outer_cache(name, spec, train, fit_rows, pred_rows, context, frozen_hash):
    signature = hashlib.sha256((frozen_hash + json.dumps({
        "name": name, "spec": spec, "fit_rows": fit_rows.tolist(),
        "predict_rows": pred_rows.tolist(), "context": context}, sort_keys=True)).encode()).hexdigest()
    path = ROOT / "experiments/v5/cv" / f"{context}_{name}_{signature[:12]}.npz"
    assert path.exists(), f"Required existing cache missing: {path}"
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["signature"]) == signature, path
        assert np.array_equal(saved["ids"], train.Id.iloc[pred_rows].to_numpy()), path
        assert np.array_equal(saved["fit_ids"], train.Id.iloc[fit_rows].to_numpy()), path
        p = saved["prediction"].copy()
    assert p.shape == (len(pred_rows),) and np.isfinite(p).all(), path
    return p, {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "signature": signature}


def outer_reference(seed, train, frozen, recipe, audit):
    rows = np.arange(len(train))
    result = pd.DataFrame({"Id": train.Id, "ylog": np.log(train.SalePrice.to_numpy()),
                           "fold": -1, "v4_log": np.nan, "v5_log": np.nan,
                           "svr_log": np.nan, "support_gate": np.nan})
    old_outer = pd.read_csv(ROOT / f"experiments/v5/outer/seed{seed}/predictions.csv")
    assert np.array_equal(old_outer.Id, train.Id)
    assert np.allclose(old_outer.log_target, result.ylog, rtol=0, atol=1e-13)
    fold_audit = []
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(rows)):
        assert not set(tr).intersection(va)
        assert np.all(old_outer.fold.iloc[va].to_numpy() == fold)
        context = f"outer{seed}_fold{fold}_outer"
        predictions, caches = {}, []
        for name in BASE_NAMES + [SVR_NAME]:
            predictions[name], record = outer_cache(name, frozen["pool"][name], train,
                tr, va, context, frozen["source_hash"])
            caches.append(record)
        v4 = np.mean([predictions[n] for n in BASE_NAMES], axis=0)
        assert np.allclose(v4, old_outer.v4.iloc[va], rtol=0, atol=1e-12)
        gate, quantiles = support_weights(train.iloc[tr], train.iloc[va])
        assert np.allclose(gate, old_outer.support_gate.iloc[va], rtol=0, atol=1e-12)
        v5 = v4 + recipe["new_weight"] * gate * (predictions[SVR_NAME] - v4)
        result.loc[va, ["fold", "v4_log", "v5_log", "svr_log", "support_gate"]] = np.column_stack(
            [np.full(len(va), fold), v4, v5, predictions[SVR_NAME], gate])
        fold_audit.append({"fold": fold, "training_rows": len(tr), "validation_rows": len(va),
                           "area_q95_q99": quantiles, "caches": caches})
    assert result.fold.between(0, 4).all() and np.isfinite(result.to_numpy()).all()
    result["fold"] = result.fold.astype(int)
    path = ART / f"full1460_seed{seed}.csv"
    result.to_csv(path, index=False)
    audit["references"].append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "rows": len(result), "seed": seed, "folds": fold_audit,
        "evaluation_role": "Retrospective fixed final-v5 recipe; not v5 automatic inner selector; reused historical data, not new independent validation"})
    return result


def original_development_reference(train, recipe, audit):
    split = pd.read_csv(ROOT / "versions/v1/artifacts/splits.csv")
    assert np.array_equal(split.Id, train.Id)
    dev_ids = pd.read_csv(ROOT / "versions/v1/artifacts/development_oof.csv").Id.to_numpy()
    id_to_index = pd.Series(np.arange(len(train)), index=train.Id)
    dev = id_to_index.loc[dev_ids].to_numpy()
    assert len(dev) == 1168 and set(dev) == set(np.flatnonzero(split.partition == "development"))
    X = train.iloc[dev]
    y = np.log(X.SalePrice.to_numpy())
    folds = np.full(len(dev), -1)
    gates = np.full(len(dev), np.nan)
    fold_audit = []
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(dev)):
        folds[va] = fold
        gates[va], quantiles = support_weights(X.iloc[tr], X.iloc[va])
        fold_audit.append({"fold": fold, "training_rows": len(tr), "validation_rows": len(va),
                           "area_q95_q99": quantiles})
    specs = read_json(ROOT / "experiments/v3/specifications.json")
    old_hash = source_hash(V3_SOURCES)
    assert old_hash == read_json(ROOT / "experiments/v3/selection.json")["source_hash"]
    predictions, caches = {}, []
    for name in BASE_NAMES:
        signature = hashlib.sha256((old_hash + json.dumps({"name": name, "spec": specs[name],
            "seed": 42, "rows": dev.tolist(), "prefix": "dev"}, sort_keys=True)).encode()).hexdigest()
        path = ROOT / f"experiments/v3/cv/dev_{name}_seed42.npz"
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved["signature"]) == signature, path
            assert np.array_equal(saved["ids"], dev_ids), path
            assert np.array_equal(saved["fold_ids"], folds), path
            predictions[name] = saved["prediction"].copy()
        caches.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "signature": signature})
    kernel_specs = read_json(ROOT / "experiments/v5/kernel_screen/specifications.json")
    hasher = hashlib.sha256()
    for source in KERNEL_SOURCES:
        hasher.update((ROOT / source).read_bytes())
    hasher.update(json.dumps({"spec": kernel_specs[SVR_NAME], "seed": 42,
        "dev_indices": dev.tolist(), "folds": 5}, sort_keys=True).encode())
    signature = hasher.hexdigest()
    path = ROOT / f"experiments/v5/kernel_screen/cv/{SVR_NAME}__seed42.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["signature"]) == signature, path
        assert np.array_equal(saved["ids"], dev_ids), path
        assert np.array_equal(saved["fold_ids"], folds), path
        svr = saved["prediction"].copy()
    caches.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "signature": signature})
    v4 = np.mean([predictions[n] for n in BASE_NAMES], axis=0)
    v5 = v4 + recipe["new_weight"] * gates * (svr - v4)
    result = pd.DataFrame({"Id": dev_ids, "ylog": y, "fold": folds, "v4_log": v4,
                           "v5_log": v5, "svr_log": svr, "support_gate": gates})
    assert np.isfinite(result.to_numpy()).all()
    path = ART / "original_dev1168_seed42.csv"
    result.to_csv(path, index=False)
    audit["references"].append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "rows": len(result), "seed": 42, "folds": fold_audit, "caches": caches,
        "evaluation_role": "Historical original development split; screening comparison only, no new independent validation",
        "baseline_bagging_caveat": "Original v3 dev caches use one CatBoost seed per fold, rather than full v5 outer three-seed bagging. New SVR is deterministic. All comparisons must use this matched reference."})
    return result


def main():
    ART.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/raw/train.csv", keep_default_na=False, na_values=["NA"])
    assert len(train) == 1460 and train.Id.is_unique and np.isfinite(train.SalePrice).all()
    frozen_path = ROOT / "experiments/v5/frozen_candidates.json"
    frozen = read_json(frozen_path)
    assert frozen["source_hash"] == source_hash(V5_SOURCES), "Historical v5 source or dependencies changed"
    assert frozen["protocol_sha256"] == sha(ROOT / "experiments/v5/protocol.json")
    selection = read_json(ROOT / "experiments/v5/selection.json")
    recipe = selection["recipe"]
    assert recipe == {"new_names": [SVR_NAME], "new_weight": .3, "support_gate": True}
    assert frozen["baseline_names"] == BASE_NAMES
    assert frozen["pool"][SVR_NAME]["spec"]["params"] == {"C": 5., "gamma": .001, "epsilon": .01}
    audit = {"method": "Cache-only reconstruction; no fits or recipe selection performed",
        "fixed_v5_recipe": recipe, "baseline_component_weights": {n: .25 for n in BASE_NAMES},
        "formula": "v5_log=v4_log+0.3*fold_training_area_gate*(svr_log-v4_log)",
        "source_hash_v5": frozen["source_hash"], "frozen_candidates_sha256": sha(frozen_path),
        "selection_sha256": sha(ROOT / "experiments/v5/selection.json"),
        "train_sha256": sha(ROOT / "data/raw/train.csv"), "script_sha256": sha(Path(__file__)),
        "validation_caveat": "v5 final recipe was selected after historical screening. Applying it to old outer folds is retrospective fixed-recipe analysis, not the previously reported nested selector evaluation and not fresh independent evidence.",
        "references": []}
    references = [(f"full1460_seed{s}", outer_reference(s, train, frozen, recipe, audit)) for s in [5501, 5509]]
    references.append(("original_dev1168_seed42", original_development_reference(train, recipe, audit)))
    metrics = []
    for name, frame in references:
        for model in ["v4_log", "v5_log"]:
            metrics.append({"reference": name, "model": model,
                "log_rmse": float(np.sqrt(np.mean((frame[model] - frame.ylog)**2)))})
    pd.DataFrame(metrics).to_csv(ART / "summary.csv", index=False)
    write_json(ART / "audit.json", audit)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print("Cache-only references written to", ART, flush=True)


if __name__ == "__main__":
    main()
