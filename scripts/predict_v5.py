"""Reproduce v5 from its frozen v4 base and saved new estimators."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from scripts.run_experiments import load_data,validate_submission


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    art=ROOT/"experiments/v5"
    manifest=json.loads((art/"model_manifest.json").read_text())
    audit=json.loads((art/"submission_audit.json").read_text())
    _,test=load_data()
    for name,expected in audit["previous_files_preserved"].items():assert digest(ROOT/name)==expected
    baseline=ROOT/manifest["baseline_submission"]
    assert digest(baseline)==manifest["baseline_sha256"]
    base=pd.read_csv(baseline);validate_submission(base,test);base_log=np.log(base.SalePrice.to_numpy())
    new_log=np.zeros(len(test))
    with threadpool_limits(limits=2):
        for entry in manifest["new_models"]:
            path=ROOT/entry["path"];assert digest(path)==entry["sha256"]
            new_log+=entry["weight_in_new_component"]*joblib.load(path).predict(test)
    if manifest["recipe"]["support_gate"]:
        q95,q99=manifest["gate_area_q95_q99"]
        gate=np.clip((q99-test.GrLivArea.to_numpy())/(q99-q95),0.,1.) if q99>q95 else (test.GrLivArea.to_numpy()<=q95).astype(float)
    else:gate=np.ones(len(test))
    prediction=base_log+manifest["recipe"]["new_weight"]*gate*(new_log-base_log)
    frame=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(prediction)});validate_submission(frame,test)
    expected=pd.read_csv(ROOT/"submission_v5.csv")
    assert digest(ROOT/"submission_v5.csv")==audit["sha256"]
    assert np.array_equal(frame.Id,expected.Id)
    np.testing.assert_allclose(frame.SalePrice,expected.SalePrice,rtol=1e-12,atol=1e-8)
    frame.to_csv(art/"reproduced_submission_v5.csv",index=False)
    result={"reproduced_rows":len(frame),"max_abs_log_prediction_difference":float(np.max(np.abs(np.log(frame.SalePrice)-np.log(expected.SalePrice)))),
        "previous_submission_files_preserved":True,"all_model_file_hashes_verified":True,
        "comparison_rtol":1e-12,"generated_file":"experiments/v5/reproduced_submission_v5.csv"}
    (art/"reproduction_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
