# 版本与线上结果

| 版本 | Git 标签 | 本地开发集 5 折 log-RMSE | 预留集 log-RMSE | Kaggle Public Score | 状态 |
|---|---|---:|---:|---:|---|
| v1 | `v1.0.0` | 0.12186 | 0.10630 | 0.11969 | 用户已提交 |
| v2 | `v2.0.0` | 0.11973（3 次 5 折均值） | 0.10038（旧预留集，仅诊断） | **0.11741** | 用户已提交 |
| v3 | `v3.0.0` | 0.11680（原 3 个拆分）；0.11941（共 5 个拆分） | 0.09778（旧预留集，仅诊断） | 待提交 | 新文件已生成 |

首版方案为 75% CatBoost + 25% ElasticNet，融合在自然对数空间完成。线上分数由用户反馈，未通过 Kaggle 接口独立核验。

v2 比较 44 个模型配置，完成 60 次交叉验证（300 次折内拟合），再比较固定权重融合。最终为 50% 新特征 ElasticNet + 50% CatBoost，后者对 3 个完整训练种子的模型取平均。新文件为根目录的 `submission_v2.csv`。

2026-10-15，用户反馈 v2 线上分数 **0.11741**，较 v1 的 0.11969 降低 0.00228，约 **1.90%**。该结果作为单独的分数记录提交到 Git；`v2.0.0` 标签和训练产物保持冻结。

v3 从 34 个配置开始，测试稀有类别缩放、街区与面积区间组合、Box–Cox、样条和新的树模型设置。最终采用 50% `en_inter_size_scaled` + 50% `cat_d5_onehot`，CatBoost 取三个全量训练种子的平均，生成 `submission_v3.csv`。

v3 在与 v2 相同的原三个拆分上均改善，平均从 0.11973 降至 0.11680。冻结参数后，在 2701、901 两个新增拆分上检查，均改善，平均从 0.12551 降至 0.12333。将五个拆分统一比较，v2 为 **0.12204**，v3 为 **0.11941**，约降低 **2.16%**。这五个拆分使用同一批开发数据，后两个不用于重新调参；不能把这些本地分数当作线上分数。

公平的重复验证比较：在相同 1,168 条开发样本、相同 3 个 5 折拆分上，v1 平均为 **0.12220**，v2 为 **0.11973**，三个拆分均改善。表中 v1 的 0.12186 是首版单个拆分的结果，不能直接与 v2 的重复均值混用。旧的 292 条预留样本已被 v1 评价，在 v2 中只作诊断，不用于选择方案。

## 保存原则

- `submission.csv` 保留首版原文件。新版本使用 `submission_v2.csv` 等明确名称。
- `versions/v1/` 保存首版的代码、数据、模型、参数、逐折预测、指标和提交文件。
- `versions/index.json` 汇总版本、文件 SHA-256、验证分数和用户反馈的线上分数。
- 每个完成的候选版本使用独立实验目录，保存拆分、运行配置、预测和评价；不复用参数不同的缓存。
- 每个完成版本提交到 `main`，并用独立 Git 标签冻结。标签对应的完整代码和产物可以通过 Git 检出。
- Kaggle 的提交由用户完成。记录线上结果时注明用户反馈来源，不把本地验证分数当作线上分数。

v2 的完整实验配置、重复验证预测、模型和报告保存在 `experiments/v2/`。`versions/v2/` 另存提交文件及关键元数据，完整源代码由 `v2.0.0` 标签冻结。

v3 的完整实验、检查和模型保存在 `experiments/v3/`，`versions/v3/` 保存提交文件和关键元数据，源代码及训练产物由 `v3.0.0` 标签冻结。v1、v2 的 CSV 校验值保持不变。

## 上传范围

这个私有仓库保存项目代码、官方原始 CSV 和 ZIP、训练后的模型、实验缓存、报告、图表、日志与提交文件。

`.venv/`、`.runtime/`、`.cache/`、Python 字节码和 macOS `.DS_Store` 是本机环境与缓存，不纳入 Git。Python 环境由 `requirements.lock.txt` 复现；OpenMP 库由 `scripts/setup_openmp.py` 安装。GitHub CLI 是本机上传工具，凭据保留在系统钥匙串，不写入项目。

## 复现首版

当前目录的模型可以直接复现 `submission.csv`：

```sh
sh scripts/run.sh scripts/predict.py
```

也可以用当前根目录的已安装环境运行保存的首版脚本：

```sh
sh scripts/run.sh versions/v1/scripts/predict.py
```

若要复现首版全部实验，在独立目录检出 `v1.0.0`，按 README 创建 Python 3.12 环境后执行实验脚本。

## 复现 v2

```sh
# 从已保存模型生成相同预测，并检查 v1 原文件未改动。
sh scripts/run.sh scripts/predict_v2.py

# 重跑同一套实验，参数与源代码相同的结果缓存会经过校验后复用。
sh scripts/run.sh scripts/optimize_v2.py screen
sh scripts/run.sh scripts/optimize_v2.py confirm
sh scripts/run.sh scripts/optimize_v2.py finalize
sh scripts/run.sh scripts/report_v2.py
```

筛选与重复验证均只使用原开发集。重新拟合最终模型时使用所有 1,460 条官方训练样本。用户已手动提交 v2，并反馈线上分数 0.11741；模型脚本不会自动向 Kaggle 提交。

## 复现 v3

```sh
sh scripts/run.sh scripts/predict_v3.py

# 复现实验：前三个拆分选方案，冻结后检查两个新拆分。
sh scripts/run.sh scripts/optimize_v3.py screen
sh scripts/run.sh scripts/optimize_v3.py confirm
sh scripts/run.sh scripts/optimize_v3.py check
sh scripts/run.sh scripts/optimize_v3.py finalize
sh scripts/run.sh scripts/report_v3.py
```
