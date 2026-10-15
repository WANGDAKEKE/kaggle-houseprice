# House Prices — 本地实验

**Built with PriorLabs-TabPFN** · TabPFN v2 预训练模型来源与许可见 `experiments/v6/pretrained/`。

使用用户提供的官方 `house-prices-advanced-regression-techniques.zip`。本地完成模型实验及预测文件生成，用户随后在 Kaggle 提交首版并反馈 Public Score **0.11969**（2026-10-15）。指标为自然对数房价的 RMSE，越小越好。

## 输出

- `submission_v5.csv`：保留 v4、加入 30% SVR，并用折内面积分位数降低大面积外推权重。用户反馈线上 **0.11595**，目前已提交最佳版本。
- `experiments/v5/report.html`：24 个基础模型/特征配置、固定候选后的嵌套拟合比较、普通成交及年份/面积诊断。
- `submission_v4.csv`：v2/v3 对数空间等权融合，用户反馈线上 **0.11679**，是 v5 提交前的最佳版本。提交前公式估算约 0.11680。
- `experiments/v4/report.html`：v3 退步诊断、组件消融、分组对比及融合分数推算。
- `submission_v3.csv`：用户反馈线上 **0.11770**，较 v2 退步。
- `experiments/v3/report.html`：v3 类别正则化、非线性特征和新增拆分检查。
- `submission_v2.csv`：v2 提交文件；用户反馈线上分数 **0.11741**。
- `experiments/v2/report.html`：v2 的 44 个配置、重复验证比较与提交审计。
- `submission.csv`：首版最终方案，对话中关联的线上分数为 0.11969。
- `submission_baseline.csv`：Ridge 基线预测，供对照。
- `artifacts/report.html`：实验报告、模型对比和误差图。
- `artifacts/experiments.csv`：全部候选方案的开发集 5 折指标。
- `artifacts/selection.json`：按开发集选择的最终方案及固定融合权重。
- `artifacts/holdout_metrics.json`：预留检查集上的一次性评价。
- `artifacts/kaggle_results.json`：用户反馈的线上分数、对应方案与提交文件校验值。
- `artifacts/models/`：用全部官方训练样本拟合的最终模型。
- `data/manifest.json`：官方 ZIP 和每个原始文件的 SHA-256 校验值。

## 验证方法

固定 seed=42，按房价十分位分层留出 20%（292 条）作为最终检查集。其余 80%（1,168 条）用于固定的 5 折交叉验证。候选包括 Ridge、随机森林、Lasso、ElasticNet、Huber 梯度提升、CatBoost、XGBoost、LightGBM 和少量固定权重融合。

缺失值填补、偏态列选择、标准化及独热编码只在每折训练数据上拟合。MSSubClass 按类别处理，Id 排除。原始 `None` 字符串保留为有效类别，`NA` 作为缺失值。所有验证样本及训练样本均保留；未使用外部测试标签或验证集早停。v1–v3 的参数和权重由本地验证选择；v4 明确使用 v2/v3 的线上分数反馈判断固定融合的价值。

按照开发集 OOF log-RMSE 选择最终方案。在距最低值 0.0003 以内的方案中，先优先选择模型数量更少的方案，再按误差排序。这一规则在查看预留集前应用，避免为了极小的开发集分差增加模型复杂度。选定后再一次性检查预留集，不根据预留结果修改参数。随后在全部 1,460 条训练数据上重新拟合。融合在对数空间进行，再通过 exp 转回美元。

开发集最优指标存在模型选择偏差；预留分数也只对应一个固定拆分。它们都不是线上分数。首版开发集 CV 为 0.12186，预留检查集为 0.10630，用户反馈的线上 Public Score 为 0.11969。线上分数按对话关联推荐的 submission.csv，未通过 Kaggle 接口独立核验。

## 当前环境复现

在项目目录执行：

```sh
# 已安装并锁定依赖；macOS 的 OpenMP 库也放在项目内。
sh scripts/run.sh scripts/download_data.py --zip house-prices-advanced-regression-techniques.zip
sh scripts/run.sh scripts/run_experiments.py compare
sh scripts/run.sh scripts/run_experiments.py finalize
sh scripts/run.sh scripts/make_report.py

# 从保存的模型重新生成预测，并与 submission.csv 核对。
sh scripts/run.sh scripts/predict.py
```

比较阶段会使用 `artifacts/cv/*.npz` 的结果缓存。若更改代码、数据、参数或拆分，先把旧 `artifacts` 目录移到新的归档位置，再重新运行；不能复用原缓存。

若重新创建环境，使用 Python 3.12，并安装 `requirements.lock.txt` 中的依赖。Apple Silicon 的 XGBoost/LightGBM 需要 OpenMP：运行 `python scripts/setup_openmp.py`，随后用 `sh scripts/run.sh ...` 启动。该脚本下载 R 项目分发的已签名 LLVM OpenMP 库并检查校验值，仅写入 `.runtime/`，不修改系统安装。

## 项目结构

```text
src/house_prices.py           特征、预处理和模型定义
scripts/download_data.py     官方 ZIP 解压或 kagglehub 下载
scripts/run_experiments.py   5 折比较、预留集检查、全量训练
scripts/predict.py           从已保存模型复现提交文件
scripts/make_report.py       数据审计、静态图表和 HTML 报告
scripts/setup_openmp.py      macOS 本地运行库准备
scripts/run.sh               运行环境入口
requirements.lock.txt        实际使用的 Python 依赖版本
data/raw/                    官方原始数据
artifacts/                   实验、模型、验证和图表
```

比赛：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## 版本管理

私有仓库：[WANGDAKEKE/kaggle-houseprice](https://github.com/WANGDAKEKE/kaggle-houseprice)。版本、标签和线上结果见 [VERSIONS.md](VERSIONS.md) 与 `versions/index.json`。首版冻结为 `v1.0.0`，完整快照保存在 `versions/v1/`；后续提交文件使用独立版本名。

v2 冻结为 `v2.0.0`。在与首版相同的三个 5 折拆分上，平均开发集 log-RMSE 从 0.12220 降至 0.11973。最终采用 50% 新特征 ElasticNet + 50% CatBoost（3 个训练种子平均）。用户反馈 `submission_v2.csv` 的线上分数为 **0.11741**，较 v1 的 0.11969 降低约 1.90%。

```sh
sh scripts/run.sh scripts/predict_v2.py
```

v3 冻结为 `v3.0.0`，采用 50% 新特征 ElasticNet + 50% 调整类别处理的 CatBoost（3 个训练种子平均）。相同的原三个拆分均值从 v2 的 0.11973 降至 0.11680；冻结方案后增加两个拆分检查，两个也都改善。五个拆分统一对比，v2 平均为 **0.12204**，v3 为 **0.11941**。用户反馈线上为 **0.11770**，反而较 v2 高 0.00029。复查发现，Id 524、1299 两套极端房屋贡献了净开发集平方误差改善的 110.7%；其他房屋合计略有退步。新增随机拆分仍使用相同房屋，不能视为独立验证。

```sh
sh scripts/run.sh scripts/predict_v3.py
```

v4 冻结为 `v4.0.0`，直接计算两版价格的几何平均，不新增训练。官方排行榜可见说明为使用全部测试数据，因此可用两个已知 log-RMSE 和两份对数预测的均方距离推算融合误差。固定 50/50 融合估算 **0.11679519**，五位小数输入分数的舍入范围约 0.11679016–0.11680023；仅为公式估算，要求用户反馈分数与保存文件对应，且评分数据和指标一致。用户随后反馈实测 **0.11679**，较此前最佳 v2 的 0.11741 降低 0.00062，约 0.53%；v4 当时成为已提交最佳版本。详细前提见 `experiments/v4/score_estimate.json`。

```sh
# 校验旧版文件、复用已保存折内预测完成诊断、复现 v4 提交文件和报告。
sh scripts/run.sh scripts/optimize_v4.py
```

## v5：新基础模型与普通房屋的收益

本轮初筛 24 个配置：12 个核/线性模型、10 个树/残差模型、2 个折内目标编码方案。多数候选不改善普通房屋，已保留负结果。CatBoost MAE 虽改善总体却损害普通房屋，因此淘汰。冻结两种 SVR 和一个交叉预测残差树，比较 15 个包含原 v4 的配方。

候选、权重、门槛和代码在外层运行前提交到 Git（`389df4a`）。在全部 1,460 套房屋上做两次 5 折外层比较，每个外层训练部分重新做 3 折内层选择。两次平均：全样本从 **0.11697 降到 0.11652**，去两套异常的普通房屋诊断从 **0.10738 降到 0.10700**，Normal 成交从 **0.08877 降到 0.08745**，十个外层折的 Normal 指标均改善。约 57% 的 Normal 房屋误差下降，去掉五笔最大正收益后仍有净收益。

收益仍不稳定：seed5501 的全样本和普通房屋汇总退步；Abnorml 成交、大面积组及 2007 年组也退步。全样本和去两套异常诊断的描述性 bootstrap 区间包含零。去掉五笔最大正收益后，全体与普通房屋的净收益均转负，仍有收益集中；只有 Normal 组在这项检查后仍改善。两次拆分仍使用同一批房屋，且全局初筛看过其中 1,168 条，不能把这个结果称作整个模型搜索流程的无偏独立验证。外层分数评价自动选择流程，并非最终固定配方的独立 OOF 分数。

最终在全量训练数据内部按同一规则选定 `svr_compact_g001_C5_w0.3_support`：普通面积范围内对数预测为 70% v4 + 30% 紧凑特征 SVR；训练面积 q95–q99 区间逐步降低新增权重，q99 以上沿用 v4。面积保护是初筛后、外层前冻结的启发式，阈值来自各次训练输入，所有验证行保留。全部 1,460 条训练行用于最终模型。新文件 `submission_v5.csv` 共 1,459 行，模型重载预测完全一致，v1–v4 文件保持原 SHA-256。用户随后反馈线上分数 **0.11595**，较此前最佳 v4 的 **0.11679** 降低 **0.00084（约 0.72%）**，目前是实际提交最佳成绩。分数另存 `experiments/v5/online_result.json`，冻结方案、CSV 和 `v5.0.0` 标签保持不变。

```sh
# 从已保存模型与冻结 v4 文件复现提交文件，并检查旧版本未改动。
sh scripts/run.sh scripts/predict_v5.py

# 按冻结代码复现全部比较；缓存核验代码、数据、参数和训练/预测行指纹。
sh scripts/run.sh scripts/optimize_v5.py freeze
sh scripts/run.sh scripts/optimize_v5.py outer --seed 5501
sh scripts/run.sh scripts/optimize_v5.py outer --seed 5509
sh scripts/run.sh scripts/optimize_v5.py finalize
sh scripts/run.sh scripts/report_v5.py
```

公开方案的来源与取舍见 `experiments/v5/research_notes.md`。本轮未使用外部 Ames 真实售价、公开答案提交或按 Id 查表覆盖。代码、全部实验、模型和预测由 `v5.0.0` 标签冻结。

用户提供的同学提交历史截图显示：旧的 Lasso + SVR + CatBoost + XGBoost 等权融合为 **0.11963**，后续描述为“50% TabPFN + 50% 原集成”的文件为 **0.11390**。当前 v5 与其差 **0.00205**。截图提供了可继续研究的模型方向，但未提供代码、TabPFN 版本或控制其他条件的消融，不能据此把全部收益归因于 TabPFN。来源说明与截图文字保存在 v5 线上结果记录中。
