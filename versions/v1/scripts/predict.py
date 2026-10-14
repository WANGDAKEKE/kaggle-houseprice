"""Recreate submission.csv from saved final models, without retraining."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scripts.run_experiments import validate_submission

test = pd.read_csv(ROOT / "data/raw/test.csv", keep_default_na=False, na_values=["NA"])
selection = json.loads((ROOT / "artifacts/selection.json").read_text())
prediction = np.zeros(len(test))
for name, weight in selection["weights"].items():
    model = joblib.load(ROOT / "artifacts/models" / f"{name}.joblib")
    prediction += weight * model.predict(test)
output = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(prediction)})
validate_submission(output, test)
destination = ROOT / "artifacts/reproduced_submission.csv"
output.to_csv(destination, index=False)
reference = pd.read_csv(ROOT / "submission.csv")
assert np.array_equal(output.Id, reference.Id)
assert np.allclose(output.SalePrice, reference.SalePrice, rtol=1e-12)
print("Saved models reproduce submission.csv:", destination)
