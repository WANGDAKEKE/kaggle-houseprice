"""Focused TabPFN experiment; reuse frozen old-model predictions throughout."""
from pathlib import Path
import sys,json,hashlib,time,argparse,gc,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.house_prices_v6 import LocalTabPFN,CHECKPOINT_SHA256
from scripts.run_experiments import load_data,validate_submission

ART=ROOT/"experiments/v6"


def save(name,d):
    (ART/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n")


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y,p,frame):
    masks={"all":np.ones(len(y),bool),"ordinary":~((frame.GrLivArea.to_numpy()>4000)&(frame.SalePrice.to_numpy()<300000)),
        "normal":frame.SaleCondition.eq("Normal").to_numpy()}
    return {n:float(np.sqrt(np.mean((p[m]-y[m])**2))) for n,m in masks.items()}


def device_name():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def probe():
    import torch
    train,_=load_data();y=np.log(train.SalePrice.to_numpy())
    tr,va=next(KFold(5,shuffle=True,random_state=5509).split(train))
    device=device_name();start=time.time()
    model=LocalTabPFN(n_estimators=1,device=device).fit(train.iloc[tr],y[tr]);fit_seconds=time.time()-start
    start=time.time();small=model.predict(train.iloc[va[:32]]);small_seconds=time.time()-start
    start=time.time();p=model.predict(train.iloc[va]);predict_seconds=time.time()-start
    ref=pd.read_csv(ART/"reference/full1460_seed5509.csv")
    details={"device":device,"torch_version":torch.__version__,"mps_built":torch.backends.mps.is_built(),
        "mps_available":torch.backends.mps.is_available(),"n_estimators":1,"training_rows":len(tr),"validation_rows":len(va),
        "fit_seconds":fit_seconds,"predict_32_seconds":small_seconds,"predict_292_seconds":predict_seconds,
        "tabpfn_single_estimator_scores":metrics(y[va],p,train.iloc[va]),
        "v5_same_fold_scores":metrics(y[va],ref.v5_log.to_numpy()[va],train.iloc[va]),
        "half_log_blend_scores":metrics(y[va],.5*p+.5*ref.v5_log.to_numpy()[va],train.iloc[va]),
        "used_for_hyperparameter_selection":False,"role":"runtime and feasibility probe; not independent validation"}
    np.savez_compressed(ART/"probe_predictions.npz",ids=train.Id.iloc[va].to_numpy(),prediction=p)
    save("probe.json",details);print(json.dumps(details,indent=2),flush=True)


def cv(seed,features,estimators):
    train,_=load_data();y=np.log(train.SalePrice.to_numpy());ref=pd.read_csv(ART/"reference"/f"full1460_seed{seed}.csv")
    assert np.array_equal(ref.Id,train.Id)
    signature=hashlib.sha256((digest(ROOT/"src/house_prices_v6.py")+digest(ROOT/"data/raw/train.csv")+
        digest(ROOT/"requirements-tabpfn.lock.txt")+json.dumps({"seed":seed,"features":features,"estimators":estimators,
        "device":device_name(),"checkpoint":CHECKPOINT_SHA256},sort_keys=True)).encode()).hexdigest()
    path=ART/"cv"/f"tabpfn_v2_{features}_e{estimators}_seed{seed}.npz";path.parent.mkdir(exist_ok=True)
    p=np.full(len(train),np.nan);fold_ids=np.full(len(train),-1);elapsed=[]
    if path.exists():
        old=np.load(path)
        if str(old["signature"])==signature:p=old["prediction"];fold_ids=old["fold_ids"]
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(train)):
        assert np.array_equal(ref.fold.to_numpy()[va],np.full(len(va),fold))
        if np.isfinite(p[va]).all():continue
        start=time.time();model=LocalTabPFN(estimators,features,device_name(),42+fold).fit(train.iloc[tr],y[tr]);p[va]=model.predict(train.iloc[va]);fold_ids[va]=fold
        elapsed.append(time.time()-start)
        np.savez_compressed(path,prediction=p,fold_ids=fold_ids,ids=train.Id.to_numpy(),signature=signature,seconds=sum(elapsed))
        print(json.dumps({"seed":seed,"features":features,"estimators":estimators,"fold":fold,"seconds":elapsed[-1],
            "tabpfn":metrics(y[va],p[va],train.iloc[va]),"v5":metrics(y[va],ref.v5_log.to_numpy()[va],train.iloc[va]),
            "half_blend":metrics(y[va],.5*(p[va]+ref.v5_log.to_numpy()[va]),train.iloc[va])}),flush=True)
        del model;gc.collect()
        import torch
        if device_name()=="mps":torch.mps.empty_cache()
    assert np.isfinite(p).all()
    records=[]
    for weight in [0.,.25,.5,.75,1.]:
        z=(1-weight)*ref.v5_log.to_numpy()+weight*p
        records.append({"name":f"log_blend_{weight}","tabpfn_weight":weight,"space":"log",**metrics(y,z,train)})
    z=np.log(.5*np.exp(ref.v5_log.to_numpy())+.5*np.exp(p))
    records.append({"name":"price_blend_0.5","tabpfn_weight":.5,"space":"price",**metrics(y,z,train)})
    table=pd.DataFrame(records);table.to_csv(ART/f"comparison_{features}_e{estimators}_seed{seed}.csv",index=False)
    pd.DataFrame({"Id":train.Id,"ylog":y,"fold":fold_ids,"v5_log":ref.v5_log,"tabpfn_log":p}).to_csv(ART/f"oof_{features}_e{estimators}_seed{seed}.csv",index=False)
    print(table.to_string(index=False),flush=True)


def finalize(seed,features,estimators):
    train,test=load_data();y=np.log(train.SalePrice.to_numpy())
    comparison=pd.read_csv(ART/f"comparison_{features}_e{estimators}_seed{seed}.csv")
    baseline=comparison[comparison.name=="log_blend_0.0"].iloc[0]
    # Fixed 50/50 log-space blend reproduces the screenshot's broad idea. We do
    # not select a fine weight grid or blend space using the same reported CV.
    candidate=comparison[comparison.name=="log_blend_0.5"].iloc[0]
    if not (candidate.normal<baseline.normal and candidate.ordinary<baseline.ordinary):
        raise RuntimeError("Half blend fails ordinary/Normal comparison; preserve negative result instead of claiming improvement")
    base=pd.read_csv(ROOT/"submission_v5.csv");validate_submission(base,test)
    base_hash=digest(ROOT/"submission_v5.csv")
    start=time.time();model=LocalTabPFN(estimators,features,device_name(),42).fit(train,y);tab=model.predict(test)
    pred=.5*np.log(base.SalePrice.to_numpy())+.5*tab
    frame=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(pred)});validate_submission(frame,test)
    frame.to_csv(ROOT/"submission_v6.csv",index=False)
    pd.DataFrame({"Id":test.Id,"tabpfn_log":tab,"v5_log":np.log(base.SalePrice),"v6_log":pred}).to_csv(ART/"test_component_log_predictions.csv",index=False)
    import joblib
    # Save only data-specific context and exact settings; the original public
    # 44MB weights are stored once alongside their own license, not duplicated.
    joblib.dump({"train":train,"ylog":y,"features":features,"estimators":estimators,"seed":42},ART/"fit_context.joblib")
    selection={"version":"v6","model_name":"TabPFN-v2_v5_half_log_blend","public_log_rmse":None,
        "model_version":"TabPFN v2","screenshot_tabpfn_version_known":False,"features":features,"n_estimators":estimators,
        "device":device_name(),"weights_in_log_space":{"v5":.5,"TabPFN_v2":.5},"comparison_seed":seed,
        "validation_role":"matched historical full-train KFold; not new independent data; no old model refitting",
        "local_half_blend_metrics":{n:float(candidate[n]) for n in ["all","ordinary","normal"]},
        "v5_same_local_metrics":{n:float(baseline[n]) for n in ["all","ordinary","normal"]},
        "pretrained_sha256":CHECKPOINT_SHA256,"base_submission_sha256":base_hash,
        "pipeline_source_sha256":digest(ROOT/"src/house_prices_v6.py"),"elapsed_final_fit_predict_seconds":time.time()-start}
    save("selection.json",selection)
    audit={"rows":len(frame),"id_order_matches_test":True,"positive_finite":True,"sha256":digest(ROOT/"submission_v6.csv"),
        "preserved_previous_submission_sha256":{n:digest(ROOT/n) for n in ["submission.csv","submission_v2.csv","submission_v3.csv","submission_v4.csv","submission_v5.csv"]},
        "submitted_to_kaggle":False,"min_price":float(frame.SalePrice.min()),"max_price":float(frame.SalePrice.max())}
    save("submission_audit.json",audit);print(json.dumps({"selection":selection,"audit":audit},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["probe","cv","finalize"])
    parser.add_argument("--seed",type=int,choices=[5501,5509],default=5509);parser.add_argument("--features",choices=["raw","physical"],default="raw")
    parser.add_argument("--estimators",type=int,default=4);args=parser.parse_args()
    if args.stage=="probe":probe()
    elif args.stage=="cv":cv(args.seed,args.features,args.estimators)
    else:finalize(args.seed,args.features,args.estimators)
