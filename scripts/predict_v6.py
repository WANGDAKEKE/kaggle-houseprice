"""Recreate the TabPFN v6 blend from its exact in-context training data.

Built with PriorLabs-TabPFN. Foundation weights are stored once with LICENSE.txt.
"""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from src.house_prices_v6 import LocalTabPFN
from scripts.optimize_v6 import device_name
from scripts.run_experiments import load_data,validate_submission


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    art=ROOT/"experiments/v6";selection=json.loads((art/"selection.json").read_text());audit=json.loads((art/"submission_audit.json").read_text())
    assert digest(ROOT/"src/house_prices_v6.py")==selection["pipeline_source_sha256"]
    for field,path in {"feature_source_sha256":"src/house_prices.py","requirements_sha256":"requirements-tabpfn.lock.txt",
                       "train_sha256":"data/raw/train.csv","test_sha256":"data/raw/test.csv"}.items():
        assert digest(ROOT/path)==selection[field],path
    for name,expected in audit["preserved_previous_submission_sha256"].items():assert digest(ROOT/name)==expected
    assert digest(ROOT/"submission_v6.csv")==audit["sha256"]
    if "fit_context_sha256" in selection:
        assert digest(art/"fit_context.joblib")==selection["fit_context_sha256"]
    context=joblib.load(art/"fit_context.joblib");_,test=load_data();start=time.time()
    if "components" in context:
        from scripts.tabpfn_variants_v6 import feature_frame
        assert digest(ROOT/"scripts/tabpfn_variants_v6.py")==selection["quality_transform_source_sha256"]
        assert context["weights_in_log_space"]==selection["weights_in_log_space"]
        predictions=[]
        for part in context["components"]:
            X=part["train"];T=test
            if part.get("feature_variant")=="quality_categorical":
                X=feature_frame(X,"raw_categorical_quality_e4")
                T=feature_frame(T,"raw_categorical_quality_e4")
            model=LocalTabPFN(part["estimators"],part["features"],device_name(),part["seed"]).fit(X,part["ylog"])
            predictions.append(model.predict(T))
            del model
            import gc,torch
            gc.collect()
            if device_name()=="mps":torch.mps.empty_cache()
        p=np.mean(predictions,axis=0)
    else:
        model=LocalTabPFN(context["estimators"],context["features"],device_name(),context["seed"]).fit(context["train"],context["ylog"])
        p=model.predict(test)
    base=pd.read_csv(ROOT/"submission_v5.csv");validate_submission(base,test)
    base_weight=selection["weights_in_log_space"]["v5"]
    blended=base_weight*np.log(base.SalePrice.to_numpy())+(1-base_weight)*p
    output=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(blended)});validate_submission(output,test)
    existing=pd.read_csv(ROOT/"submission_v6.csv");max_difference=float(np.max(np.abs(blended-np.log(existing.SalePrice.to_numpy()))))
    # MPS floating point may vary slightly; do not claim bitwise reproducibility.
    assert max_difference<5e-5,max_difference
    output.to_csv(art/"reproduced_submission_v6.csv",index=False)
    result={"rows":len(output),"max_abs_log_prediction_difference":max_difference,
        "log_prediction_tolerance":5e-5,"device":device_name(),"elapsed_seconds":time.time()-start,
        "previous_files_preserved":True,"checkpoint_revalidated":True,"bitwise_identical_prices":bool(np.array_equal(output.SalePrice.to_numpy(),existing.SalePrice.to_numpy()))}
    (art/"reproduction_audit.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
