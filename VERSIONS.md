# 版本与线上结果

| 版本 | Git 标签 | 本地开发集 5 折 log-RMSE | 预留集 log-RMSE | Kaggle Public Score | 状态 |
|---|---|---:|---:|---:|---|
| v1 | `v1.0.0` | 0.12186 | 0.10630 | 0.11969 | 用户已提交 |

首版方案为 75% CatBoost + 25% ElasticNet，融合在自然对数空间完成。线上分数由用户反馈，未通过 Kaggle 接口独立核验。

## 保存原则

- `submission.csv` 保留首版原文件。新版本使用 `submission_v2.csv` 等明确名称。
- `versions/v1/` 保存首版的代码、数据、模型、参数、逐折预测、指标和提交文件。
- `versions/index.json` 汇总版本、文件 SHA-256、验证分数和用户反馈的线上分数。
- 每个完成的候选版本使用独立实验目录，保存拆分、运行配置、预测和评价；不复用参数不同的缓存。
- 每个完成版本提交到 `main`，并用独立 Git 标签冻结。标签对应的完整代码和产物可以通过 Git 检出。
- Kaggle 的提交由用户完成。记录线上结果时注明用户反馈来源，不把本地验证分数当作线上分数。

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
