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
    for name,expected in audit["preserved_previous_submission_sha256"].items():assert digest(ROOT/name)==expected
    assert digest(ROOT/"submission_v6.csv")==audit["sha256"]
    context=joblib.load(art/"fit_context.joblib");_,test=load_data();start=time.time()
    model=LocalTabPFN(context["estimators"],context["features"],device_name(),context["seed"]).fit(context["train"],context["ylog"])
    p=model.predict(test);base=pd.read_csv(ROOT/"submission_v5.csv");validate_submission(base,test)
    blended=.5*np.log(base.SalePrice.to_numpy())+.5*p
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
