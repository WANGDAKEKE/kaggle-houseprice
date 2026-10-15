"""Frozen shortlist, nested fit evaluation, and feature-dependent v4 grafts.

Historical screening is disclosed: this is not an unbiased evaluation of every
model-development decision ever made on these houses. All per-fit transforms,
gates and inner recipe selections exclude the outer validation labels.
"""
from pathlib import Path
import sys, os, json, hashlib, argparse, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.house_prices import Features, CatFrame
from src.house_prices_v3 import build_model as build_old
from src.house_prices_v5_kernel import build_model as build_kernel
from src.house_prices_v5_trees import build_model as build_tree
from src.house_prices_v5_encoding import build_model as build_encoding
from scripts.run_experiments import load_data, validate_submission

ART=ROOT/"experiments/v5"
SEEDS=[42,2026,137]
SOURCE_FILES=["scripts/optimize_v5.py","src/house_prices.py","src/house_prices_v2.py",
    "src/house_prices_v3.py","src/house_prices_v5_kernel.py","src/house_prices_v5_trees.py",
    "src/house_prices_v5_encoding.py","requirements.lock.txt","data/raw/train.csv"]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash():return hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def fit_seeds(spec):
    stochastic=spec["family"]=="tree" or (spec["family"]=="baseline" and spec["spec"]["kind"]=="catboost")
    return SEEDS if stochastic else [42]


def masks(frame):
    extreme=(frame.GrLivArea.to_numpy()>4000)&(frame.SalePrice.to_numpy()<300000)
    return {"all":np.ones(len(frame),bool),"ordinary":~extreme,
            "normal":frame.SaleCondition.eq("Normal").to_numpy()}


def metrics(y,p,frame):
    return {name:float(np.sqrt(np.mean((p[m]-y[m])**2))) for name,m in masks(frame).items()}


def support_weights(train_frame,predict_frame):
    q95,q99=np.quantile(train_frame.GrLivArea,[.95,.99])
    area=predict_frame.GrLivArea.to_numpy()
    if q99<=q95:
        return (area<=q95).astype(float)
    return np.clip((q99-area)/(q99-q95),0.,1.)


class FoldCat(RegressorMixin,BaseEstimator):
    """Frozen v4 CatBoost semantics, with two CPU threads per fit."""
    def __init__(self,params=None,seed=42):self.params=params;self.seed=seed
    def fit(self,X,y):
        from catboost import CatBoostRegressor
        self.model_=CatBoostRegressor(loss_function="RMSE",random_seed=self.seed,thread_count=2,
            verbose=False,allow_writing_files=False,cat_features=list(X.select_dtypes(exclude="number").columns),**self.params)
        self.model_.fit(X,y)
        return self
    def predict(self,X):return self.model_.predict(X)


def build(name,spec,seed):
    if spec["family"]=="baseline":
        old=spec["spec"]
        if old["kind"]=="catboost":
            params=old.get("reference_spec",old)["params"]
            return Pipeline([("features",Features(engineered=True,log_numeric=False)),
                ("frame",CatFrame()),("model",FoldCat(params,seed))])
        return build_old(old,seed)
    return {"kernel":build_kernel,"tree":build_tree,"encoding":build_encoding}[spec["family"]](spec["spec"],seed)


def freeze():
    protocol=json.loads((ART/"protocol.json").read_text())
    old=json.loads((ROOT/"experiments/v3/specifications.json").read_text())
    base_names=["ref_v2_linear","ref_v2_cat","en_inter_size_scaled","cat_d5_onehot"]
    pool={n:{"family":"baseline","spec":old[n]} for n in base_names}
    kernel_specs=json.loads((ART/"kernel_screen/specifications.json").read_text())
    kernel=pd.read_csv(ART/"kernel_screen/screen_results.csv")
    for metric in ["without_extremes_diagnostic","normal_sales"]:
        kernel[f"{metric}_gain"]=kernel[f"{metric}_v4_reference_rmse"]-kernel[f"{metric}_quarter_graft_rmse"]
    kernel=kernel[(kernel.without_extremes_diagnostic_gain>0)&(kernel.normal_sales_gain>0)].copy()
    kernel["gain"]=kernel.without_extremes_diagnostic_gain+kernel.normal_sales_gain
    for n in kernel.sort_values("gain",ascending=False).model.head(2):
        pool[n]={"family":"kernel","spec":kernel_specs[n]}
    tree_specs=json.loads((ART/"tree_screen/specifications.json").read_text())
    tree=pd.read_csv(ART/"tree_screen/results.csv")
    table=tree[tree.method=="graft025_v4"].pivot(index="name",columns="group",values="log_rmse")
    ref=tree[tree.method=="graft025_v4"].pivot(index="name",columns="group",values="v4_log_rmse")
    eligible=table[(table.without_two_diagnostic<ref.without_two_diagnostic)&(table.normal_sales<ref.normal_sales)].copy()
    if len(eligible):
        eligible["gain"]=(ref.without_two_diagnostic-eligible.without_two_diagnostic)+(ref.normal_sales-eligible.normal_sales)
        for n in eligible.sort_values("gain",ascending=False).index[:2]:
            pool[n]={"family":"tree","spec":tree_specs[n]}
    enc_specs=json.loads((ART/"encoding_screen/specifications.json").read_text())
    enc=pd.read_csv(ART/"encoding_screen/results.csv");ref_enc=enc[enc.model=="v4"].iloc[0]
    eligible_enc=enc[(enc.new_weight==.15)&(enc.ordinary<ref_enc.ordinary)&(enc.normal<ref_enc.normal)]
    for n in eligible_enc.model.head(2):pool[n]={"family":"encoding","spec":enc_specs[n]}
    new_names=[n for n in pool if n not in base_names]
    groups={n:[n] for n in new_names}
    if len(new_names)>1:groups["new_equal"]=new_names
    recipes={"v4":{"new_names":[],"new_weight":0.,"support_gate":False}}
    for label,names in groups.items():
        gate_options=[False,True] if any(pool[n]["family"]=="kernel" for n in names) else [False]
        for weight in protocol["recipe_grid"]["new_weight"]:
            for gated in gate_options:
                recipes[f"{label}_w{weight}_{'support' if gated else 'uniform'}"]={
                    "new_names":names,"new_weight":weight,"support_gate":gated}
    frozen={"pool":pool,"baseline_names":base_names,"new_names":new_names,"recipes":recipes,
        "source_hash":source_hash(),"protocol_sha256":digest(ART/"protocol.json"),
        "support_gate_revision":"After historical screening, before any new outer results: compare uniform graft with a fixed feature-only q95-to-q99 area taper. Quantiles fitted on each actual fit subset; no target/Id branch.",
        "baseline_bagging":"Each baseline CatBoost fit averages seeds42/2026/137, matching v4 bagging; linear fits once. Two CPU threads do not change the configured model.",
        "outer_used_for_shortlisting":False}
    write_json(ART/"frozen_candidates.json",frozen)
    print(json.dumps(frozen,indent=2),flush=True)


def load_frozen():
    frozen=json.loads((ART/"frozen_candidates.json").read_text())
    assert frozen["source_hash"]==source_hash(),"Source changed after freezing candidates"
    assert frozen["protocol_sha256"]==digest(ART/"protocol.json")
    return frozen


def fit_predict(name,spec,train,y,fit_rows,predict_rows,context):
    signature=hashlib.sha256((source_hash()+json.dumps({"name":name,"spec":spec,
        "fit_rows":fit_rows.tolist(),"predict_rows":predict_rows.tolist(),"context":context},sort_keys=True)).encode()).hexdigest()
    path=ART/"cv"/f"{context}_{name}_{signature[:12]}.npz"
    if path.exists():
        saved=np.load(path)
        assert str(saved["signature"])==signature
        assert np.array_equal(saved["ids"],train.Id.iloc[predict_rows])
        assert np.array_equal(saved["fit_ids"],train.Id.iloc[fit_rows])
        return saved["prediction"]
    assert not set(fit_rows).intersection(predict_rows)
    start=time.time();p=np.zeros(len(predict_rows))
    seeds=fit_seeds(spec)
    with threadpool_limits(limits=2):
        for seed in seeds:
            model=build(name,spec,seed).fit(train.iloc[fit_rows],y[fit_rows])
            p+=model.predict(train.iloc[predict_rows])/len(seeds)
    assert np.isfinite(p).all()
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,prediction=p,ids=train.Id.iloc[predict_rows].to_numpy(),
        fit_ids=train.Id.iloc[fit_rows].to_numpy(),signature=signature,seconds=time.time()-start,
        actual_model_fits=len(seeds))
    print(f"{context} {name}: {len(seeds)} fits, {time.time()-start:.1f}s",flush=True)
    return p


def recipe_prediction(recipe,predictions,gate,base_names):
    baseline=sum(predictions[n] for n in base_names)/len(base_names)
    if not recipe["new_names"]:return baseline
    new=sum(predictions[n] for n in recipe["new_names"])/len(recipe["new_names"])
    support=gate if recipe["support_gate"] else 1.
    return baseline+recipe["new_weight"]*support*(new-baseline)


def choose(frozen,predictions,gate,y,frame):
    reference=metrics(y,recipe_prediction(frozen["recipes"]["v4"],predictions,gate,frozen["baseline_names"]),frame)
    records=[]
    for name,recipe in frozen["recipes"].items():
        p=recipe_prediction(recipe,predictions,gate,frozen["baseline_names"])
        values=metrics(y,p,frame)
        eligible=values["all"]<=reference["all"]+.0005 and values["ordinary"]<=reference["ordinary"]-.00015 and values["normal"]<=reference["normal"]-.00015
        objective=.5*values["all"]+.25*values["ordinary"]+.25*values["normal"]
        records.append({"name":name,**values,"eligible":eligible,"objective":objective,
            "weight":recipe["new_weight"],"new_components":len(recipe["new_names"])})
    table=pd.DataFrame(records);eligible=table[table.eligible]
    if len(eligible):
        near=eligible[eligible.objective<=eligible.objective.min()+.00005]
        selected=near.sort_values(["weight","new_components","objective","name"]).iloc[0]["name"]
    else:selected="v4"
    return selected,table


def inner_predictions(frozen,train,y,rows,seed,context):
    predictions={n:np.full(len(rows),np.nan) for n in frozen["pool"]}
    gates=np.full(len(rows),np.nan);folds=np.full(len(rows),-1)
    for fold,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=seed).split(rows)):
        fit_rows,pred_rows=rows[tr],rows[va]
        for n,spec in frozen["pool"].items():
            predictions[n][va]=fit_predict(n,spec,train,y,fit_rows,pred_rows,f"{context}_inner{fold}")
        gates[va]=support_weights(train.iloc[fit_rows],train.iloc[pred_rows]);folds[va]=fold
    assert np.isfinite(gates).all() and all(np.isfinite(p).all() for p in predictions.values())
    return predictions,gates,folds


def outer(seed):
    frozen=load_frozen();train,_=load_data();y=np.log(train.SalePrice.to_numpy());rows=np.arange(len(train))
    path=ART/"outer"/f"seed{seed}";path.mkdir(parents=True,exist_ok=True)
    all_base=np.full(len(train),np.nan);all_selected=np.full(len(train),np.nan);gates=np.full(len(train),np.nan);folds=np.full(len(train),-1)
    records=[]
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(rows)):
        context=f"outer{seed}_fold{fold}"
        inner,g,inner_folds=inner_predictions(frozen,train,y,tr,19000+seed+fold,context)
        selected,table=choose(frozen,inner,g,y[tr],train.iloc[tr])
        table.to_csv(path/f"fold{fold}_inner_recipes.csv",index=False)
        pd.DataFrame({"Id":train.Id.iloc[tr],"inner_fold":inner_folds,"support_gate":g}).to_csv(path/f"fold{fold}_inner_assignments.csv",index=False)
        predictions={n:fit_predict(n,spec,train,y,tr,va,f"{context}_outer") for n,spec in frozen["pool"].items()}
        gate=support_weights(train.iloc[tr],train.iloc[va])
        baseline=recipe_prediction(frozen["recipes"]["v4"],predictions,gate,frozen["baseline_names"])
        p=recipe_prediction(frozen["recipes"][selected],predictions,gate,frozen["baseline_names"])
        all_base[va]=baseline;all_selected[va]=p;gates[va]=gate;folds[va]=fold
        baseline_scores=metrics(y[va],baseline,train.iloc[va]);selected_scores=metrics(y[va],p,train.iloc[va])
        record={"fold":fold,"selected_recipe":selected,"v4":baseline_scores,"selected":selected_scores,
            "outer_train_area_q95_q99":np.quantile(train.GrLivArea.iloc[tr],[.95,.99]).tolist(),
            "validation_support_tapered_rows":int(np.sum(gate<1)),"validation_support_zero_rows":int(np.sum(gate==0))}
        records.append(record);write_json(path/"fold_results.json",records)
        print("OUTER RESULT",seed,json.dumps(record),flush=True)
    assert np.isfinite(all_base).all() and np.isfinite(all_selected).all() and (folds>=0).all()
    pd.DataFrame({"Id":train.Id,"log_target":y,"fold":folds,"v4":all_base,"selected":all_selected,"support_gate":gates}).to_csv(path/"predictions.csv",index=False)
    summary={"seed":seed,"v4":metrics(y,all_base,train),"selected":metrics(y,all_selected,train),
        "folds":records,"source_hash":frozen["source_hash"],"protocol_sha256":frozen["protocol_sha256"],
        "frozen_candidates_sha256":digest(ART/"frozen_candidates.json"),"predictions_sha256":digest(path/"predictions.csv"),
        "evaluation_role":"fixed screened shortlist nested-fit evaluation; historical screening and data reuse remain"}
    write_json(path/"summary.json",summary)
    print("OUTER SUMMARY",json.dumps(summary),flush=True)


def finalize():
    frozen=load_frozen();protocol=json.loads((ART/"protocol.json").read_text())
    summaries=[json.loads((ART/"outer"/f"seed{s}"/"summary.json").read_text()) for s in protocol["outer_cv"]["seeds"]]
    for seed,summary in zip(protocol["outer_cv"]["seeds"],summaries):
        assert summary["seed"]==seed and len(summary["folds"])==5
        assert sorted(r["fold"] for r in summary["folds"])==list(range(5))
        assert summary["source_hash"]==frozen["source_hash"]
        assert summary["protocol_sha256"]==frozen["protocol_sha256"]
        assert summary["frozen_candidates_sha256"]==digest(ART/"frozen_candidates.json")
        assert summary["predictions_sha256"]==digest(ART/"outer"/f"seed{seed}"/"predictions.csv")
    means={kind:{metric:float(np.mean([r[kind][metric] for r in summaries])) for metric in ["all","ordinary","normal"]} for kind in ["v4","selected"]}
    passed=means["selected"]["all"]<=means["v4"]["all"]+.0003 and all(means["selected"][m]<means["v4"][m] for m in ["ordinary","normal"])
    gate_result={"passed":passed,"mean_two_outer_repeat_scores":means,"criterion":protocol["release_gate"],"retuned_after_outer_results":False}
    write_json(ART/"release_gate.json",gate_result)
    if not passed:raise RuntimeError("Predeclared release gate failed; preserve all results, do not claim a validated improvement")
    train,test=load_data();y=np.log(train.SalePrice.to_numpy());rows=np.arange(len(train))
    predictions,gates,folds=inner_predictions(frozen,train,y,rows,24501,"final_selection")
    selected,table=choose(frozen,predictions,gates,y,train);table.to_csv(ART/"final_inner_recipes.csv",index=False)
    recipe=frozen["recipes"][selected]
    if selected=="v4":raise RuntimeError("Final selection falls back to v4; no distinct candidate to release")
    pd.DataFrame({"Id":train.Id,"inner_fold":folds,"support_gate":gates}).to_csv(ART/"final_inner_assignments.csv",index=False)
    for name,expected in protocol["frozen_submission_hashes"].items():assert digest(ROOT/name)==expected
    base=pd.read_csv(ROOT/"submission_v4.csv");validate_submission(base,test);base_log=np.log(base.SalePrice.to_numpy())
    models=[];new_predictions={};(ART/"models").mkdir(exist_ok=True)
    for name in recipe["new_names"]:
        spec=frozen["pool"][name];seeds=fit_seeds(spec)
        pred=np.zeros(len(test))
        for seed in seeds:
            with threadpool_limits(limits=2):
                model=build(name,spec,seed).fit(train,y);pred+=model.predict(test)/len(seeds)
            model_path=ART/"models"/f"{name}_seed{seed}.joblib";joblib.dump(model,model_path)
            models.append({"name":name,"seed":seed,"weight_in_new_component":1/len(recipe["new_names"])/len(seeds),
                "path":str(model_path.relative_to(ROOT)),"sha256":digest(model_path)})
        new_predictions[name]=pred
    new_log=sum(new_predictions.values())/len(new_predictions)
    support=support_weights(train,test) if recipe["support_gate"] else np.ones(len(test))
    out_log=base_log+recipe["new_weight"]*support*(new_log-base_log)
    submission=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(out_log)})
    validate_submission(submission,test);destination=ROOT/"submission_v5.csv";submission.to_csv(destination,index=False)
    pd.DataFrame({"Id":test.Id,"v4_log_prediction":base_log,"new_log_prediction":new_log,"support_weight":support,
        **{f"new_{n}":v for n,v in new_predictions.items()},"v5_log_prediction":out_log}).to_csv(ART/"test_component_predictions.csv",index=False)
    manifest={"baseline_submission":"submission_v4.csv","baseline_sha256":digest(ROOT/"submission_v4.csv"),
        "new_models":models,"recipe":recipe,"gate_area_q95_q99":np.quantile(train.GrLivArea,[.95,.99]).tolist(),
        "gate_formula":"clip((q99-GrLivArea)/(q99-q95),0,1); gate=1 for uniform recipe",
        "source_hash":frozen["source_hash"]}
    write_json(ART/"model_manifest.json",manifest)
    selection={"version":"v5","selected_recipe":selected,"recipe":recipe,"final_inner_selection_seed":24501,
        "final_inner_scores":table[table.name==selected].iloc[0].to_dict(),"public_log_rmse":None,
        "v4_user_reported_public_log_rmse":.11679,"outer_release_gate":gate_result,
        "outer_results_used_for_recipe_selection":False,"source_hash":frozen["source_hash"],
        "validation_caveat":"Screening saw historical development data before shortlist freeze. Nested inner selection excludes each outer validation fold but cannot undo that historical reuse."}
    write_json(ART/"selection.json",selection)
    audit={"rows":len(submission),"sha256":digest(destination),"positive_finite":True,"id_order_matches_test":True,
        "previous_files_preserved":protocol["frozen_submission_hashes"],"min_price":float(submission.SalePrice.min()),
        "max_price":float(submission.SalePrice.max()),"submitted_to_kaggle":False,
        "test_rows_support_tapered":int(np.sum(support<1)),"test_rows_support_zero":int(np.sum(support==0)),
        "price_ratio_to_v4_quantiles":{str(q):float(np.quantile(submission.SalePrice/base.SalePrice,q)) for q in [0,.01,.5,.99,1]}}
    write_json(ART/"submission_audit.json",audit)
    snapshot=ROOT/"versions/v5";snapshot.mkdir(exist_ok=True)
    for name in ["selection.json","model_manifest.json","submission_audit.json","release_gate.json","protocol.json","frozen_candidates.json"]:
        (snapshot/name).write_bytes((ART/name).read_bytes())
    (snapshot/"submission_v5.csv").write_bytes(destination.read_bytes())
    print("FINAL SELECTION",json.dumps(selection),flush=True);print("SUBMISSION AUDIT",json.dumps(audit),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["freeze","outer","finalize"]);parser.add_argument("--seed",type=int,choices=[5501,5509]);args=parser.parse_args()
    if args.stage=="outer":
        assert args.seed is not None
        outer(args.seed)
    elif args.stage=="freeze":freeze()
    else:finalize()
