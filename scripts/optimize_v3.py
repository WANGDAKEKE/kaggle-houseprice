"""Version 3 experiments, frozen v2 comparison, new-split checks and predictions."""
from pathlib import Path
import sys, os, json, time, hashlib, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from src.house_prices import log_rmse
from src.house_prices_v3 import build_model
from scripts.optimize_v2 import partitions, cache_signature as v2_signature
from scripts.run_experiments import load_data, validate_submission

ART = ROOT / "experiments/v3"
SEEDS = [42, 2026, 137]
CHECK_SEEDS = [2701, 901]
SOURCE_HASH = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in [
    "src/house_prices.py", "src/house_prices_v2.py", "src/house_prices_v3.py",
    "scripts/optimize_v3.py", "requirements.lock.txt", "data/raw/train.csv"])).hexdigest()


def specs():
    old = json.loads((ROOT / "experiments/v2/specifications.json").read_text())
    config = {
        "ref_v2_linear": {"reference": "en_rich_a0.0007_l0.8", "reference_spec": old["en_rich_a0.0007_l0.8"], "kind": "elasticnet"},
        "ref_v2_cat": {"reference": "v1_catboost", "reference_spec": old["v1_catboost"], "kind": "catboost"},
    }
    def en(name, **kw):
        config[name] = {"kind": "elasticnet", "params": {"alpha": .0007, "l1_ratio": .8}, **kw}
    for scale in [.15, .25, .5]:
        en(f"en_catscale_{scale}", category_scale=scale)
    for scale in [0, .25]:
        for power in [.15, .3]:
            en(f"en_boxcox_{power}_catscale_{scale}", power="boxcox", power_lambda=power, category_scale=scale)
    for inter in ["sale", "size", "both"]:
        en(f"en_inter_{inter}", interactions=inter)
        en(f"en_inter_{inter}_scaled", interactions=inter, category_scale=.25)
    en("en_quality_categories", quality_categories=True)
    en("en_quality_categories_scaled", quality_categories=True, category_scale=.25)
    for knots in [3, 5]:
        en(f"en_splines_{knots}", splines=True, n_knots=knots)
        en(f"en_splines_{knots}_scaled", splines=True, n_knots=knots, category_scale=.25)
    for alpha in [.0003, .0012]:
        config[f"en_spline_alpha_{alpha}"] = {"kind": "elasticnet", "splines": True, "n_knots": 4,
            "params": {"alpha": alpha, "l1_ratio": .8}}
    for leaves, reg in [(7, 5), (15, 5), (15, 15)]:
        config[f"histgb_leaf{leaves}_l{reg}"] = {"kind": "histgb", "features": "v1", "params": {
            "max_iter": 1000, "learning_rate": .025, "max_leaf_nodes": leaves,
            "min_samples_leaf": 15, "l2_regularization": reg}}
    for name, params in [
        ("cat_d4_long", {"depth": 4, "iterations": 3500, "learning_rate": .03, "l2_leaf_reg": 5}),
        ("cat_d5_regularized", {"depth": 5, "iterations": 3000, "learning_rate": .025, "l2_leaf_reg": 10}),
        ("cat_d5_ctr2", {"depth": 5, "iterations": 2200, "learning_rate": .035, "l2_leaf_reg": 5, "max_ctr_complexity": 2}),
        ("cat_d5_onehot", {"depth": 5, "iterations": 2500, "learning_rate": .03, "l2_leaf_reg": 5, "one_hot_max_size": 20}),
        ("cat_d4_huber", {"depth": 4, "iterations": 2600, "learning_rate": .03, "l2_leaf_reg": 5, "loss_function": "Huber:delta=0.12"}),
        ("cat_d5_huber", {"depth": 5, "iterations": 2600, "learning_rate": .03, "l2_leaf_reg": 5, "loss_function": "Huber:delta=0.15"}),
        ("cat_d4_ordered", {"depth": 4, "iterations": 1800, "learning_rate": .04, "l2_leaf_reg": 5, "boosting_type": "Ordered"}),
        ("cat_d5_lowrandom", {"depth": 5, "iterations": 2600, "learning_rate": .03, "l2_leaf_reg": 5, "random_strength": .2, "bagging_temperature": .3}),
    ]:
        config[name] = {"kind": "catboost", "features": "v1", "params": params}
    return config


def cv(name, spec, seed, train, y, rows, prefix="dev"):
    signature = hashlib.sha256((SOURCE_HASH + json.dumps({"name": name, "spec": spec, "seed": seed, "rows": rows.tolist(), "prefix": prefix}, sort_keys=True)).encode()).hexdigest()
    path = ART / "cv" / f"{prefix}_{name}_seed{seed}.npz"
    if path.exists():
        saved = np.load(path)
        if str(saved["signature"]) == signature:
            return saved["prediction"], float(saved["seconds"])
    reused = False
    if prefix == "dev" and spec.get("reference") and seed in SEEDS:
        source = ROOT / "experiments/v2/cv" / f"{spec['reference']}__seed{seed}.npz"
        saved = np.load(source)
        assert str(saved["signature"]) == v2_signature(spec["reference"], spec["reference_spec"], seed, rows)
        assert np.array_equal(saved["dev_ids"], train.Id.iloc[rows])
        prediction = saved["prediction"]
        fold_ids = saved["fold_ids"]
        seconds = 0.
        reused = True
    else:
        start = time.time()
        prediction = np.full(len(rows), np.nan)
        fold_ids = np.full(len(rows), -1)
        for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(rows)):
            model = build_model(spec, seed + fold)
            model.fit(train.iloc[rows[tr]], y[rows[tr]])
            prediction[va] = model.predict(train.iloc[rows[va]])
            fold_ids[va] = fold
            print(f"{prefix} {name} seed={seed} fold={fold+1}: {log_rmse(y[rows[va]], prediction[va]):.5f}", flush=True)
        seconds = time.time() - start
    assert np.isfinite(prediction).all()
    np.savez_compressed(path, prediction=prediction, fold_ids=fold_ids, ids=train.Id.iloc[rows].to_numpy(),
        signature=signature, seconds=seconds, reused_v2=reused)
    return prediction, seconds


def screen():
    train, _ = load_data(); y = np.log(train.SalePrice.to_numpy()); dev, _ = partitions(train)
    config = specs(); (ART / "specifications.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for name, spec in config.items():
        p, seconds = cv(name, spec, 42, train, y, dev)
        rows.append({"model": name, "log_rmse": log_rmse(y[dev], p), "seconds": seconds})
        pd.DataFrame(rows).sort_values("log_rmse").to_csv(ART / "screen_results.csv", index=False)
    result = pd.DataFrame(rows).sort_values("log_rmse")
    shortlist = list(result.model.head(4))
    for family in ["elasticnet", "histgb", "catboost"]:
        available = result.loc[result.model.map(lambda n: config[n]["kind"] == family)]
        name = available.iloc[0].model
        if name not in shortlist:
            shortlist.append(name)
    shortlist = list(dict.fromkeys(shortlist + ["ref_v2_linear", "ref_v2_cat"]))
    (ART / "shortlist.json").write_text(json.dumps(shortlist, indent=2) + "\n")
    print("Screen:", result.to_string(index=False), flush=True)
    print("Shortlist:", shortlist, flush=True)


def blend_scores(weights, predictions, target):
    p = sum(predictions[name] * weight for name, weight in weights.items())
    return [log_rmse(target, p[:, i]) for i in range(p.shape[1])]


def confirm():
    train, _ = load_data(); y = np.log(train.SalePrice.to_numpy()); dev, _ = partitions(train)
    config = json.loads((ART / "specifications.json").read_text())
    shortlist = json.loads((ART / "shortlist.json").read_text())
    predictions, base_rows = {}, []
    for name in shortlist:
        predictions[name] = np.column_stack([cv(name, config[name], seed, train, y, dev)[0] for seed in SEEDS])
        scores = [log_rmse(y[dev], predictions[name][:, i]) for i in range(len(SEEDS))]
        base_rows.append({"model": name, "mean_log_rmse": np.mean(scores), **{f"seed_{s}": scores[i] for i,s in enumerate(SEEDS)}})
        pd.DataFrame(base_rows).sort_values("mean_log_rmse").to_csv(ART / "confirmed_base_models.csv", index=False)
    means = {r["model"]: r["mean_log_rmse"] for r in base_rows}
    linear = min([n for n in shortlist if config[n]["kind"] == "elasticnet"], key=means.get)
    tree = min([n for n in shortlist if config[n]["kind"] in ["catboost", "histgb"]], key=means.get)
    candidates = {n: {n: 1.} for n in shortlist}
    v2 = {"ref_v2_linear": .5, "ref_v2_cat": .5}
    candidates["v2_frozen"] = v2
    def add(name, entries):
        weights = {}
        for n,w in entries:
            weights[n] = weights.get(n, 0.) + w
        candidates[name] = weights
    for weight in [.25, .5, .75]:
        add(f"best_linear_tree_{weight}", [(linear,weight),(tree,1-weight)])
    # A stronger individual predictor need not be the best blend component.
    # Compare the shortlisted linear/tree pairs on the same small fixed grid.
    linears = [n for n in shortlist if config[n]["kind"] == "elasticnet"]
    trees = [n for n in shortlist if config[n]["kind"] in ["catboost", "histgb"]]
    for linear_name in linears:
        for tree_name in trees:
            for weight in [.25, .5, .75]:
                add(f"pair_{linear_name}_{tree_name}_{weight}", [(linear_name,weight),(tree_name,1-weight)])
    # New model grafts are fixed weights, never selected using holdout or public scores.
    for name in shortlist:
        if name.startswith("ref_"):
            continue
        for weight in [.25, .5]:
            add(f"v2_plus_{name}_{weight}", [("ref_v2_linear",.5*(1-weight)),("ref_v2_cat",.5*(1-weight)),(name,weight)])
    if linear != "ref_v2_linear" or tree != "ref_v2_cat":
        add("new_linear_oldcat_half", [(linear,.5),("ref_v2_cat",.5)])
        add("oldlinear_newtree_half", [("ref_v2_linear",.5),(tree,.5)])
    records = []
    for name, weights in candidates.items():
        values = blend_scores(weights, predictions, y[dev])
        records.append({"model":name,"mean_log_rmse":np.mean(values),"components":len(weights),
            **{f"seed_{s}":values[i] for i,s in enumerate(SEEDS)}})
    result = pd.DataFrame(records).sort_values("mean_log_rmse")
    result.to_csv(ART / "confirmation_results.csv", index=False)
    close = result.loc[result.mean_log_rmse <= result.iloc[0].mean_log_rmse + .00015]
    chosen = close.sort_values(["components","mean_log_rmse"]).iloc[0]
    reference = result.loc[result.model == "v2_frozen"].iloc[0]
    selection = {"version":"v3","name":chosen.model,"weights":candidates[chosen.model],
        "mean_repeated_cv_log_rmse":float(chosen.mean_log_rmse),"v2_mean_repeated_cv_log_rmse":float(reference.mean_log_rmse),
        "seeds":SEEDS,"selected_seed_scores":{str(s):float(chosen[f"seed_{s}"]) for s in SEEDS},
        "v2_seed_scores":{str(s):float(reference[f"seed_{s}"]) for s in SEEDS},
        "selection_rule":"three known development splits; within 0.00015 prefer fewer components; frozen before new split checks",
        "check_seeds":CHECK_SEEDS,"public_score":None,"v2_user_reported_public_score":.11741,
        "validation_rows_removed":0,"holdout_used_for_selection":False,"source_hash":SOURCE_HASH}
    (ART / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(result.to_string(index=False),flush=True)
    print("Frozen selection:",json.dumps(selection),flush=True)


def check():
    train, _ = load_data(); y=np.log(train.SalePrice.to_numpy()); dev,_=partitions(train)
    config=json.loads((ART/"specifications.json").read_text());selection=json.loads((ART/"selection.json").read_text())
    selected=selection["weights"]; reference={"ref_v2_linear":.5,"ref_v2_cat":.5}
    scores=[]
    for seed in CHECK_SEEDS:
        predictions={n:cv(n,config[n],seed,train,y,dev)[0] for n in set(selected)|set(reference)}
        chosen_p=sum(predictions[n]*w for n,w in selected.items())
        old_p=sum(predictions[n]*w for n,w in reference.items())
        scores.append({"seed":seed,"v3_log_rmse":log_rmse(y[dev],chosen_p),"v2_log_rmse":log_rmse(y[dev],old_p)})
    report={"role":"additional split checks after freezing selection; same development houses, not an untouched dataset",
        "used_for_retuning":False,"scores":scores,"mean_v3":float(np.mean([x["v3_log_rmse"] for x in scores])),
        "mean_v2":float(np.mean([x["v2_log_rmse"] for x in scores]))}
    (ART/"additional_split_check.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Additional split checks:",json.dumps(report),flush=True)


def finalize():
    train,test=load_data();y=np.log(train.SalePrice.to_numpy())
    config=json.loads((ART/"specifications.json").read_text());selection=json.loads((ART/"selection.json").read_text())
    _,hold=partitions(train);dev,_=partitions(train)
    predicted=np.zeros(len(test));held=np.zeros(len(hold));entries=[]
    components=pd.DataFrame({"Id":test.Id})
    for name,weight in selection["weights"].items():
        seeds=SEEDS if config[name]["kind"] in ["catboost","histgb","gbr"] else [42]
        for seed in seeds:
            model=build_model(config[name],seed);model.fit(train.iloc[dev],y[dev]);held+=weight/len(seeds)*model.predict(train.iloc[hold])
            model=build_model(config[name],seed);model.fit(train,y);p=model.predict(test)
            predicted+=weight/len(seeds)*p
            path=ART/"models"/f"{name}_seed{seed}.joblib";joblib.dump(model,path)
            components[f"{name}_seed{seed}"]=p
            entries.append({"name":name,"seed":seed,"weight":weight/len(seeds),"path":str(path.relative_to(ROOT))})
            print("Final fit:",name,seed,flush=True)
    submission=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(predicted)})
    validate_submission(submission,test);submission.to_csv(ROOT/"submission_v3.csv",index=False)
    components.to_csv(ART/"test_component_log_predictions.csv",index=False)
    (ART/"model_manifest.json").write_text(json.dumps(entries,indent=2)+"\n")
    pd.DataFrame({"Id":train.Id.iloc[hold].to_numpy(),"log_target":y[hold],"log_prediction":held}).to_csv(ART/"old_holdout_predictions.csv",index=False)
    old=pd.read_csv(ROOT/"submission_v2.csv");assert np.array_equal(old.Id,test.Id)
    ratios=submission.SalePrice/old.SalePrice
    audit={"rows":len(submission),"positive_finite":True,"id_order_matches_test":True,
        "sha256":hashlib.sha256((ROOT/"submission_v3.csv").read_bytes()).hexdigest(),
        "v2_preserved_sha256":hashlib.sha256((ROOT/"submission_v2.csv").read_bytes()).hexdigest(),
        "min_price":float(submission.SalePrice.min()),"max_price":float(submission.SalePrice.max()),
        "price_ratio_to_v2_quantiles":{str(q):float(ratios.quantile(q)) for q in [.01,.5,.99]},
        "old_holdout_log_rmse_diagnostic_only":log_rmse(y[hold],held),"submitted_to_kaggle":False}
    (ART/"submission_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("v3 audit:",json.dumps(audit),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["screen","confirm","check","finalize","all"]);args=parser.parse_args()
    for d in [ART,ART/"cv",ART/"models"]:d.mkdir(parents=True,exist_ok=True)
    for name in ["screen","confirm","check","finalize"]:
        if args.stage in [name,"all"]:globals()[name]()
