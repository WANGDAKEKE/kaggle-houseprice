"""Reproduce v3 from fitted models and verify preservation of submitted versions."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scripts.run_experiments import validate_submission

test=pd.read_csv(ROOT/"data/raw/test.csv",keep_default_na=False,na_values=["NA"])
art=ROOT/"experiments/v3"
entries=json.loads((art/"model_manifest.json").read_text())
assert abs(sum(e["weight"] for e in entries)-1)<1e-12
prediction=np.zeros(len(test))
for entry in entries:
    model=joblib.load(ROOT/entry["path"])
    prediction+=entry["weight"]*model.predict(test)
submission=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(prediction)})
validate_submission(submission,test)
reference=pd.read_csv(ROOT/"submission_v3.csv")
assert np.array_equal(reference.Id,submission.Id)
assert np.allclose(reference.SalePrice,submission.SalePrice,rtol=1e-12)
submission.to_csv(art/"reproduced_submission_v3.csv",index=False)
index=json.loads((ROOT/"versions/index.json").read_text())
for version in index["versions"]:
    if version["version"] in ["v1","v2"]:
        assert hashlib.sha256((ROOT/version["submission_file"]).read_bytes()).hexdigest()==version["sha256"]
for name,expected in [("submission.csv","210223272bc07d8c7d483aadc47a9831faa248ca9c5c633699d12ebea9b3e3db"),
                      ("submission_v2.csv","66c2d66ab1b9fe43cfd0bc68229a9881c1181fb87115e5fad6350d363972eb6a")]:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
print("v3 saved-model predictions reproduced; v1 and v2 submissions are unchanged.")
