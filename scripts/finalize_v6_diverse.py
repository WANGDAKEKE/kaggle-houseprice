"""Fixed equal bag of physical/e16 and quality-categorical/e4 TabPFN.

Preserve the physical-only candidate before replacing the main candidate with
Equal thirds of v5, physical/e16 and quality/e4 in log-price space. Representation
choices were informed by reused historical folds: disclose selection bias.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from src.house_prices_v6 import LocalTabPFN
from scripts.tabpfn_variants_v6 import feature_frame
from scripts.run_experiments import load_data, validate_submission

ART = ROOT / "experiments/v6"
BACKUP = ART / "physical_candidate"
DETAIL = ART / "diverse"
SEED = 5509


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(y, p, frame):
    groups = {"all": np.ones(len(y), dtype=bool),
        "ordinary": ~((frame.GrLivArea.to_numpy() > 4000) & (frame.SalePrice.to_numpy() < 300000)),
        "normal": frame.SaleCondition.eq("Normal").to_numpy()}
    return {n: float(np.sqrt(np.mean((p[m]-y[m])**2))) for n, m in groups.items()}


def prepare():
    train, _ = load_data()
    physical = pd.read_csv(ART / f"oof_physical_e16_seed{SEED}.csv")
    quality = pd.read_csv(ART / "variants/raw_categorical_quality_e4_oof.csv")
    reference = pd.read_csv(ART / f"reference/full1460_seed{SEED}.csv")
    y = np.log(train.SalePrice.to_numpy())
    for frame in [physical, quality, reference]:
        assert np.array_equal(frame.Id, train.Id)
        assert np.allclose(frame.ylog, y, rtol=0, atol=1e-13)
        assert np.array_equal(frame.fold, reference.fold)
        assert np.allclose(frame.v5_log, reference.v5_log, rtol=0, atol=1e-13)
    physical_log = physical.tabpfn_log.to_numpy()
    quality_log = quality.variant_log.to_numpy()
    tabpfn_log = .5 * (physical_log + quality_log)
    baseline = reference.v5_log.to_numpy()
    assert np.isfinite(tabpfn_log).all()
    oof = pd.DataFrame({"Id": train.Id, "ylog": y, "fold": reference.fold,
        "v5_log": baseline, "tabpfn_log": tabpfn_log,
        "physical_e16_log": physical_log, "quality_e4_log": quality_log})
    oof.to_csv(ART / f"oof_diverse_e20_seed{SEED}.csv", index=False)
    records = []
    for weight in [0., .25, .5, 2./3., .75, 1.]:
        p = (1.-weight)*baseline + weight*tabpfn_log
        records.append({"name": f"log_blend_{weight}", "tabpfn_weight": weight, "space": "log",
                        **metrics(y, p, train)})
    price_p = np.log(.5*np.exp(baseline) + .5*np.exp(tabpfn_log))
    records.append({"name": "price_blend_0.5", "tabpfn_weight": .5, "space": "price",
                    **metrics(y, price_p, train)})
    comparison = pd.DataFrame(records)
    comparison.to_csv(ART / f"comparison_diverse_e20_seed{SEED}.csv", index=False)
    result = {"method": "natural equal-thirds log blend of v5, physical/e16 and quality/e4",
        "weights_in_log_space": {"v5": 1./3., "physical_e16": 1./3., "quality_e4": 1./3.},
        "comparison_seed": SEED, "all_validation_rows_retained": True,
        "physical_half_metrics": metrics(y, .5*(baseline+physical_log), train),
        "diverse_half_metrics": metrics(y, .5*(baseline+tabpfn_log), train),
        "diverse_equal_thirds_metrics": metrics(y, (baseline+physical_log+quality_log)/3., train),
        "v5_metrics": metrics(y, baseline, train),
        "selection_caveat": "Model representations and ensemble size were explored on these same historical folds; final bag is informed by those comparisons, not independent validation. No finer bag weight optimization was performed.",
        "rationale": "Choose natural equal thirds after comparing historical folds: it improves ordinary and Normal groups over the 50% baseline-weight draft and 75% TabPFN grid point. Most additional all-house gain from 75% versus 50% came from two extreme houses. Keep the initial half candidate and disclose adaptive weight selection.",
        "source_sha256": {"physical_oof": digest(ART / f"oof_physical_e16_seed{SEED}.csv"),
            "quality_oof": digest(ART / "variants/raw_categorical_quality_e4_oof.csv"),
            "v5_reference": digest(ART / f"reference/full1460_seed{SEED}.csv"),
            "quality_transform": digest(ROOT / "scripts/tabpfn_variants_v6.py"),
            "finalizer": digest(Path(__file__))}}
    json_write(DETAIL / "oof_summary.json", result)
    print(comparison.to_string(index=False), flush=True)
    return comparison, result


def backup_physical():
    items = {"selection.json": ART / "selection.json", "fit_context.joblib": ART / "fit_context.joblib",
        "test_component_log_predictions.csv": ART / "test_component_log_predictions.csv",
        "submission_v6.csv": ROOT / "submission_v6.csv", "submission_audit.json": ART / "submission_audit.json"}
    manifest_path = BACKUP / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["sha256"].items():
            assert digest(BACKUP / name) == expected, name
        return
    selection = json.loads(items["selection.json"].read_text())
    assert selection["features"] == "physical" and selection["n_estimators"] == 16
    audit = json.loads(items["submission_audit.json"].read_text())
    assert audit["sha256"] == digest(items["submission_v6.csv"])
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name, source in items.items():
        assert source.exists(), source
        if (BACKUP / name).exists():
            assert digest(BACKUP / name) == digest(source), name
        else:
            shutil.copy2(source, BACKUP / name)
    json_write(manifest_path, {"role": "Exact original physical/e16 + v5 half blend retained before representation bag",
        "sha256": {name: digest(BACKUP / name) for name in items},
        "source_paths": {name: str(path.relative_to(ROOT)) for name, path in items.items()}})


def device_name():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def finalize():
    started = time.time()
    comparison, oof_summary = prepare()
    backup_physical()
    selection = json.loads((BACKUP / "selection.json").read_text())
    old_audit = json.loads((BACKUP / "submission_audit.json").read_text())
    physical_context = joblib.load(BACKUP / "fit_context.joblib")
    physical_test = pd.read_csv(BACKUP / "test_component_log_predictions.csv")
    train, test = load_data()
    y = np.log(train.SalePrice.to_numpy())
    assert np.array_equal(physical_test.Id, test.Id)
    assert physical_context["features"] == "physical" and physical_context["estimators"] == 16
    assert np.array_equal(physical_context["train"].Id, train.Id)
    assert np.allclose(physical_context["ylog"], y, rtol=0, atol=1e-13)
    for name, expected in old_audit["preserved_previous_submission_sha256"].items():
        assert digest(ROOT / name) == expected, name
    base = pd.read_csv(ROOT / "submission_v5.csv")
    validate_submission(base, test)
    base_log = np.log(base.SalePrice.to_numpy())
    assert np.allclose(physical_test.v5_log, base_log, rtol=0, atol=1e-13)
    assert np.allclose(physical_test.v6_log, .5*(base_log+physical_test.tabpfn_log.to_numpy()), rtol=0, atol=1e-12)

    device = device_name()
    quality_context = {"train": train, "ylog": y, "features": "raw", "estimators": 4,
        "seed": 42, "feature_variant": "quality_categorical"}
    quality_settings = {"checkpoint_sha256": selection["pretrained_sha256"],
        "pipeline_source_sha256": digest(ROOT / "src/house_prices_v6.py"),
        "quality_transform_source_sha256": digest(ROOT / "scripts/tabpfn_variants_v6.py"),
        "train_sha256": digest(ROOT / "data/raw/train.csv"), "test_sha256": digest(ROOT / "data/raw/test.csv"),
        "requirements_sha256": digest(ROOT / "requirements-tabpfn.lock.txt"),
        "features": "raw", "estimators": 4, "seed": 42, "feature_variant": "quality_categorical", "device": device}
    signature = hashlib.sha256(json.dumps(quality_settings, sort_keys=True).encode()).hexdigest()
    cache = DETAIL / "quality_e4_full_test.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as old:
            assert str(old["signature"]) == signature
            assert np.array_equal(old["ids"], test.Id.to_numpy())
            quality_log = old["prediction"].copy()
            quality_elapsed = float(old["elapsed_seconds"])
    else:
        quality_started = time.time()
        model = LocalTabPFN(4, "raw", device, 42).fit(feature_frame(train, "raw_categorical_quality_e4"), y)
        for column in ["OverallQual", "OverallCond"]:
            assert model.columns_.index(column) in model.categorical_indices_
        quality_log = model.predict(feature_frame(test, "raw_categorical_quality_e4"))
        quality_elapsed = time.time()-quality_started
        np.savez_compressed(cache, ids=test.Id.to_numpy(), prediction=quality_log,
                            signature=signature, elapsed_seconds=quality_elapsed)
        del model
        gc.collect()
        if device == "mps":
            import torch
            torch.mps.empty_cache()
    assert quality_log.shape == (len(test),) and np.isfinite(quality_log).all()
    physical_log = physical_test.tabpfn_log.to_numpy()
    tabpfn_log = .5*(physical_log+quality_log)
    blended = (base_log+physical_log+quality_log)/3.
    output = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(blended)})
    validate_submission(output, test)
    output.to_csv(ROOT / "submission_v6.csv", index=False)
    pd.DataFrame({"Id": test.Id, "tabpfn_log": tabpfn_log, "v5_log": base_log, "v6_log": blended,
        "physical_e16_log": physical_log, "quality_e4_log": quality_log}).to_csv(ART / "test_component_log_predictions.csv", index=False)
    physical_component = {**physical_context, "feature_variant": "identity"}
    context = {"components": [physical_component, quality_context],
        "component_average": "equal in log target space", "baseline_submission": "submission_v5.csv",
        "weights_in_log_space": {"v5": 1./3., "physical_e16": 1./3., "quality_e4": 1./3.}}
    joblib.dump(context, ART / "fit_context.joblib")
    candidate = comparison.loc[(comparison.space=="log") & np.isclose(comparison.tabpfn_weight,2./3.)].iloc[0]
    components = [{"name": "physical_e16", "features": "physical", "estimators": 16, "seed": 42,
                   "feature_variant": "identity", "global_log_weight": 1./3.},
                  {"name": "quality_e4", "features": "raw", "estimators": 4, "seed": 42,
                   "feature_variant": "quality_categorical", "global_log_weight": 1./3.}]
    old_elapsed = float(selection["elapsed_final_fit_predict_seconds"])
    selection.update({"features": "diverse", "n_estimators": 20,
        "model_name": "TabPFN-v2_diverse_physical16_quality4_v5_equal_thirds_log_blend", "components": components,
        "weights_in_log_space": context["weights_in_log_space"], "device": device,
        "local_half_blend_metrics": {n: float(candidate[n]) for n in ["all", "ordinary", "normal"]},
        "local_selected_blend_metrics": {n: float(candidate[n]) for n in ["all", "ordinary", "normal"]},
        "legacy_metrics_key_note": "local_half_blend_metrics is retained for schema compatibility and now records the selected equal-thirds blend; use local_selected_blend_metrics.",
        "quality_transform_source_sha256": quality_settings["quality_transform_source_sha256"],
        "finalizer_source_sha256": digest(Path(__file__)),
        "elapsed_final_fit_predict_seconds": old_elapsed+quality_elapsed,
        "elapsed_diverse_finalize_wall_seconds": time.time()-started,
        "quality_full_fit_predict_seconds": quality_elapsed,
        "physical_full_fit_predict_seconds": old_elapsed,
        "submission_sha256": digest(ROOT / "submission_v6.csv"),
        "fit_context_sha256": digest(ART / "fit_context.joblib"),
        "representation_choice_rationale": oof_summary["rationale"],
        "selection_bias": oof_summary["selection_caveat"],
        "physical_candidate_backup": str(BACKUP.relative_to(ROOT)),
        "no_finer_representation_weight_search": True})
    json_write(ART / "selection.json", selection)
    audit = {**old_audit, "rows": len(output), "id_order_matches_test": True,
        "positive_finite": True, "sha256": digest(ROOT / "submission_v6.csv"),
        "preserved_previous_submission_sha256": {n: digest(ROOT / n) for n in old_audit["preserved_previous_submission_sha256"]},
        "submitted_to_kaggle": False, "min_price": float(output.SalePrice.min()),
        "max_price": float(output.SalePrice.max()), "physical_candidate_backup_manifest_sha256": digest(BACKUP / "manifest.json"),
        "quality_test_cache_sha256": digest(cache), "quality_settings": quality_settings,
        "component_rows_match_official_test": True}
    json_write(ART / "submission_audit.json", audit)
    print(json.dumps({"selection": selection, "audit": audit}, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "finalize"])
    args = parser.parse_args()
    DETAIL.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare()
    else:
        finalize()


if __name__ == "__main__":
    main()
