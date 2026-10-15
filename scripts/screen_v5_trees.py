"""Predeclared robust tree screen, fixed original dev folds, paired v4 diagnostics."""
from pathlib import Path
import os, sys, json, time, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scripts.run_experiments import load_data
from scripts.optimize_v2 import partitions
from src.house_prices_v5_trees import build_model

ART = ROOT / "experiments/v5/tree_screen"


def specifications():
    common_cat = {"iterations": 2200, "learning_rate": .035, "depth": 5,
        "l2_leaf_reg": 5, "one_hot_max_size": 20}
    common_lgb = {"n_estimators": 1800, "learning_rate": .025,
        "num_leaves": 8, "min_child_samples": 18, "colsample_bytree": .85,
        "reg_lambda": 3, "reg_alpha": .02}
    common_xgb = {"n_estimators": 1800, "learning_rate": .025,
        "max_depth": 3, "min_child_weight": 5, "subsample": .85,
        "colsample_bytree": .85, "reg_lambda": 5, "reg_alpha": .02}
    return {
        "cat_center_huber015": {"kind": "catboost", "features": "v1", "params":
            {**common_cat, "depth": 4, "loss_function": "Huber:delta=0.15"}},
        "cat_center_huber030": {"kind": "catboost", "features": "v1", "params":
            {**common_cat, "loss_function": "Huber:delta=0.30"}},
        "cat_center_mae": {"kind": "catboost", "features": "v1", "params":
            {**common_cat, "loss_function": "MAE"}},
        "cat_simple_rmse_d4": {"kind": "catboost", "params":
            {**common_cat, "depth": 4, "loss_function": "RMSE", "l2_leaf_reg": 8}},
        "lgb_huber_leaves8": {"kind": "lightgbm", "params":
            {**common_lgb, "objective": "huber", "alpha": .9}},
        "lgb_mae_leaves8": {"kind": "lightgbm", "params":
            {**common_lgb, "objective": "regression_l1"}},
        "xgb_pseudohuber015": {"kind": "xgboost", "params":
            {**common_xgb, "objective": "reg:pseudohubererror", "huber_slope": .15}},
        "xgb_square_d2_regular": {"kind": "xgboost", "params":
            {**common_xgb, "max_depth": 2, "objective": "reg:squarederror"}},
        "gbr_huber_depth2": {"kind": "gbr", "log_numeric": True, "params":
            {"loss": "huber", "alpha": .9, "n_estimators": 1500,
             "learning_rate": .03, "max_depth": 2, "min_samples_leaf": 8,
             "max_features": .8}},
        "crossfit_en_residual_cat": {"kind": "crossfit_en_residual_cat", "params":
            {"loss_function": "Huber:delta=0.15", "iterations": 1000,
             "learning_rate": .03, "depth": 3, "l2_leaf_reg": 10,
             "one_hot_max_size": 20}},
    }


def rmse(y, p):
    return float(np.sqrt(np.mean((y-p)**2)))


def main():
    ART.mkdir(parents=True, exist_ok=True)
    train, _ = load_data()
    dev, _ = partitions(train)
    ids = train.iloc[dev].Id.to_numpy()
    target = np.log(train.iloc[dev].SalePrice.to_numpy())
    names = ["ref_v2_linear", "ref_v2_cat", "en_inter_size_scaled", "cat_d5_onehot"]
    reference = [np.load(ROOT / "experiments/v3/cv" / f"dev_{n}_seed42.npz") for n in names]
    assert all(np.array_equal(r["ids"], ids) for r in reference)
    base = sum(r["prediction"] for r in reference)/4
    folds = reference[0]["fold_ids"]
    masks = {"all": np.ones(len(dev), bool),
        "without_two_diagnostic": ~np.isin(ids, [524, 1299]),
        "normal_sales": train.iloc[dev].SaleCondition.eq("Normal").to_numpy()}
    specs = specifications()
    (ART / "specifications.json").write_text(json.dumps(specs, indent=2)+"\n")
    (ART / "protocol.json").write_text(json.dumps({"split_seed":42,"n_splits":5,
        "development_rows":len(dev),"validation_rows_removed":0,"threads_per_fit":2,
        "role":"exploratory screen on repeatedly reused houses; not independent confirmation",
        "gate":"ordinary-house diagnostic and Normal sale diagnostic, alongside every validation row",
        "robust_loss_initialization":"each training fold target centered by its own mean",
        "residual_candidate":"three-fold OOF residuals generated inside each outer training fold; no in-sample residuals"},indent=2)+"\n")
    source = hashlib.sha256((ROOT/"src/house_prices_v5_trees.py").read_bytes()).hexdigest()
    rows = []
    for name, spec in specs.items():
        signature = hashlib.sha256((source+json.dumps(spec,sort_keys=True)+str(ids.tolist())).encode()).hexdigest()
        dest=ART/f"{name}_seed42.npz"
        start=time.time()
        if dest.exists() and str(np.load(dest)["signature"])==signature:
            saved=np.load(dest); pred=saved["prediction"]; seconds=float(saved["seconds"])
        else:
            pred=np.full(len(dev),np.nan)
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(dev)):
                assert np.all(folds[va]==fold)
                model=build_model(spec,42+fold)
                model.fit(train.iloc[dev[tr]],target[tr])
                pred[va]=model.predict(train.iloc[dev[va]])
                print(f"{name} fold={fold+1} RMSE={rmse(target[va],pred[va]):.6f}",flush=True)
            seconds=time.time()-start
            assert np.isfinite(pred).all()
            np.savez_compressed(dest,prediction=pred,ids=ids,fold_ids=folds,
                signature=signature,seconds=seconds)
        for method,p in [("standalone",pred),("graft025_v4",.75*base+.25*pred)]:
            for group,mask in masks.items():
                gain=(base[mask]-target[mask])**2-(p[mask]-target[mask])**2
                rows.append({"name":name,"method":method,"group":group,"rows":int(mask.sum()),
                    "log_rmse":rmse(target[mask],p[mask]),"v4_log_rmse":rmse(target[mask],base[mask]),
                    "mse_gain":float(gain.mean()),"fraction_rows_improved":float(np.mean(gain>0)),
                    "seconds":seconds})
        pd.DataFrame(rows).to_csv(ART/"results.csv",index=False)
        print(pd.DataFrame(rows[-6:]).to_string(index=False),flush=True)
    result=pd.DataFrame(rows)
    ranked=result.query("method == 'graft025_v4' and group == 'without_two_diagnostic'").sort_values("log_rmse")
    gates = result.query("method == 'graft025_v4'").pivot(index="name",columns="group",values="mse_gain")
    passing = gates.index[(gates > 0).all(axis=1)].tolist()
    ordinary_passing = gates.index[(gates[["without_two_diagnostic", "normal_sales"]] > 0).all(axis=1)].tolist()
    (ART/"recommendations.json").write_text(json.dumps({"candidate_names":passing,
        "ordinary_gate_challengers":ordinary_passing,
        "best_two_challengers_if_no_pass":ranked.name.head(2).tolist(),
        "basis":"screen only: require positive graft MSE gain in all, ordinary-house and Normal-sale groups; all validation houses retained in primary score",
        "ranked":ranked.to_dict(orient="records")},indent=2)+"\n")


if __name__ == "__main__":
    main()
