"""Reproduce the final v2 CSV from saved fitted models and verify every Id."""
from pathlib import Path
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
from scripts.run_experiments import validate_submission

ART = ROOT / "experiments/v2"
entries = json.loads((ART / "model_manifest.json").read_text())
test = pd.read_csv(ROOT / "data/raw/test.csv", keep_default_na=False, na_values=["NA"])
assert abs(sum(e["weight"] for e in entries) - 1) < 1e-12
predicted_log = np.zeros(len(test))
for entry in entries:
    path = ROOT / entry["path"]
    model = joblib.load(path)
    predicted_log += entry["weight"] * model.predict(test)
submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(predicted_log)})
validate_submission(submission, test)
original = pd.read_csv(ROOT / "submission_v2.csv")
assert np.array_equal(original.Id, submission.Id)
assert np.allclose(original.SalePrice, submission.SalePrice, rtol=1e-12)
submission.to_csv(ART / "reproduced_submission_v2.csv", index=False)
index = json.loads((ROOT / "versions/index.json").read_text())
v1 = next(v for v in index["versions"] if v["version"] == "v1")
assert hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest() == v1["sha256"]
assert hashlib.sha256((ROOT / v1["submission_file"]).read_bytes()).hexdigest() == v1["sha256"]
print("Saved v2 models reproduce submission_v2.csv; v1 submission remains unchanged.")
