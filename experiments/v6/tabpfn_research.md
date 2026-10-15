# TabPFN access and execution notes

Checked against official sources on 2026-10-15. This note records research, not an executed model benchmark.

## Immediate recommendation

Start with the publicly available **TabPFN v2 regression checkpoint**, explicitly selecting `ModelVersion.V2`. Do not instantiate the unqualified default and assume it means v2: the current package default is TabPFN-3.5. The screenshot names TabPFN but does not establish the classmate's model version, checkpoint, preprocessing, validation protocol, or whether the blend was arithmetic in price or log-price space.

The current official PyPI release is **tabpfn 9.2.0**, published 2026-10-12, Python >=3.10. Its package minimum is PyTorch >=2.5; the official README recommends PyTorch >=2.13 for best Apple Silicon performance. Current PyPI torch is **2.14.1**. TabPFN also depends on `mlx>=0.29.3` specifically on macOS ARM64. Prefer an isolated, pinned environment and preserve the earlier model environment. [PyPI release](https://pypi.org/project/tabpfn/9.2.0/), [official package metadata](https://github.com/PriorLabs/TabPFN/blob/main/pyproject.toml), [PyTorch package metadata](https://pypi.org/pypi/torch/json).

## Version and access distinctions

| Model release | Official local checkpoint | Access and scope |
| --- | --- | --- |
| v2 | `Prior-Labs/TabPFN-v2-reg` / `tabpfn-v2-regressor.ckpt` | Public model; Prior Labs License, Apache 2.0 with additional attribution. 10,000 training rows / 500 columns. Appropriate initial no-new-account route. |
| v2.5 | `Prior-Labs/tabpfn_2_5` / `tabpfn-v2.5-regressor-v2.5_default.ckpt` | Separate non-commercial model license. Current local package documents first-use Prior Labs login and license acceptance. The default regressor checkpoint is synthetic-only; `real` variants use real-data pretraining. |
| v2.6 | `Prior-Labs/tabpfn_2_6` / `tabpfn-v2.6-regressor-v2.6_default.ckpt` | Separate non-commercial license and first-use login/license acceptance. |
| v3 | `Prior-Labs/tabpfn_3` / `tabpfn-v3-regressor-v3_default.ckpt` | Separate non-commercial license and first-use login/license acceptance. |
| v3.5 / Fast | `Prior-Labs/tabpfn_3_5` / `tabpfn-v3.5-20260909.safetensors` or `tabpfn-v3.5-fast-20260909.safetensors` | Current default; one multitask checkpoint supports regression and classification. Separate non-commercial license and first-use login/license acceptance. |

Official package configuration selects these exact checkpoint names. [Model source registry](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model_loading.py), [v2 model card](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), [v2.5 model card](https://huggingface.co/Prior-Labs/tabpfn_2_5).

The modern licenses should not be accepted or falsely represented as accepted on the user's behalf. If a modern model is desired and the normal package login flow requires interaction, the user can sign in and review/accept it. A documented public v2 route avoids this dependency. Important distinction: the official Hugging Face API currently returns `gated=false` for all five model repositories above, but the current local package still documents login/license acceptance for v2.5 and later. An un-gated Hugging Face metadata field does not mean a user has accepted the model license. Login for weights does not transmit the training dataset; local inference remains local. [Official models and license FAQ](https://docs.priorlabs.ai/models).

## Official public v2 download routes

The library normally downloads v2 automatically. Its source additionally documents direct public downloads **only for v2**:

- [Official Hugging Face checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt)
- [Official source-configured fallback](https://storage.googleapis.com/tabpfn-v2-model-files/05152025/tabpfn-v2-regressor.ckpt)
- [v2 model license](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt)

The default checkpoint is **44.4 MB**; the full Hugging Face repository holds multiple variants and is much larger. The official file page publishes SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. The model repository API currently returns revision `4972a65a1b30806315c6f92499959ffbfc69a673`, so a stable URL is `https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt`. Save download URL, retrieval date, byte count and local SHA-256 before use, and pin the downloaded artifact instead of repeatedly resolving a moving `main`. Do not use a third-party mirror, use external Ames sale labels, or generalize the v2 fallback to newer license-controlled models. [Official checkpoint size/hash](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/tabpfn-v2-regressor.ckpt), [official repository metadata](https://huggingface.co/api/models/Prior-Labs/TabPFN-v2-reg).

## Mac, CPU and runtime expectations

Apple Silicon GPU execution is supported. Use explicit `device="mps"` after checking `torch.backends.mps.is_available()`, because automatic-device selection differs across package versions. v2's official model card lists **16 GB+ RAM** and permits CPU; this is a recommendation, not a tested hard memory minimum for this machine. The current general README recommends about 8 GB GPU memory for common tasks and 16 GB for some larger tasks. These statements do not provide an exact peak-memory or runtime guarantee for 1,460 x 79 House Prices data.

Our 1,460 rows and 79 raw predictors are within v2's pretrained data dimensions. Older releases' **1,000-row CPU cap** is a separate runtime guard, so a full CPU fit needs `ignore_pretraining_limits=True` or `TABPFN_ALLOW_CPU_LARGE_DATASET=true`. This is an officially documented override, not model-access bypass. Start with one estimator and one predefined fold, measure actual fit/predict seconds and memory, then consider 2 or 4 estimators. A large ensemble or a broad fold grid before this smoke test would waste time. Predict the whole validation or test batch in one call rather than one row at a time. [Hardware notes](https://github.com/PriorLabs/TabPFN#installation), [MPS regression tests](https://github.com/PriorLabs/TabPFN/blob/main/tests/test_regressor_interface.py).

No official House Prices 1,460 x 79 CPU/MPS timing was found. Any proposed minutes estimate before an on-machine smoke test would be speculation.

## Minimal current API for the first timed fold

```python
import os
os.environ["TABPFN_MODEL_CACHE_DIR"] = ".cache/tabpfn/models"
os.environ["TABPFN_NO_BROWSER"] = "true"  # fail clearly instead of unexpected login UI
os.environ["TABPFN_MPS_MEMORY_FRACTION"] = "0.7"

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

# X_train/X_valid are fold-specific raw pandas frames, without Id or SalePrice.
# Preserve the official missing/absence semantics in the existing data loader.
# Pass raw numeric and categorical columns; no one-hot expansion or scaling.
# Mark MSSubClass categorical explicitly as well as object/category columns.
device = "mps" if torch.backends.mps.is_available() else "cpu"
categorical = [
    i for i, (name, dtype) in enumerate(X_train.dtypes.items())
    if str(dtype) in {"object", "string", "category"} or name == "MSSubClass"
]
reg = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2,
    device=device,
    n_estimators=1,
    categorical_features_indices=categorical,
    random_state=42,
    n_preprocessing_jobs=1,
    inference_precision=torch.float32,
    fit_mode="fit_preprocessors",
    ignore_pretraining_limits=(device == "cpu"),
)
reg.fit(X_train, np.log(y_train_sale_price))
validation_log_prediction = reg.predict(X_valid)
```

This is an execution suggestion, not claimed tested code. Current pandas string dtype detection should be verified on the actual loaded frame (e.g. use `pandas.api.types.is_string_dtype` for robustness). For an explicit downloaded path, use the standard `TabPFNRegressor(model_path=..., ...)` constructor with the same options; the checkpoint determines the architecture. Keep target log-space consistent with Kaggle's metric.

## Comparison plan

Run the same saved folds and same retained validation rows as v5, compare TabPFN alone and prespecified 25% / 50% log-space blends with v5. Report all rows, Normal sales, rows excluding the two known extreme cheap-large homes, and per-fold changes. A single new model family that generalizes to ordinary sales is more useful than many additional near-duplicate tree hyperparameters. Do not attribute the classmate's 0.11390 to a particular unseen preprocessing detail or promise the same result. If TabPFN produces no ordinary-sale improvement, preserve the experiment as negative evidence instead of forcing a new submission.
