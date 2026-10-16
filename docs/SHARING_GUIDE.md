# House Prices 项目分享与复现说明

这个项目参加 Kaggle 的 [House Prices 比赛](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)。目前主提交是 [`submission_v6.csv`](../submission_v6.csv)，用户反馈的线上 log-RMSE 为 **0.11212**，分数越低越好。这里的成绩来自用户提交后的反馈；本地交叉验证与线上成绩分别记录。

本次分享使用固定快照 **`v6.1.0`**。之前的 `v6.0.0` 保留训练和提交时的快照，本次分享快照补充线上成绩及分享文档。

## 先看哪些文件

| 内容 | 位置 |
|---|---|
| 当前实验报告，可直接用浏览器打开 | [`experiments/v6/report.html`](../experiments/v6/report.html) |
| 可直接发送的单文件 HTML 报告 | [`docs/house_prices_v6_final_report.html`](house_prices_v6_final_report.html) |
| 可直接发送的 PDF 报告 | [`output/pdf/house_prices_v6_final_report.pdf`](../output/pdf/house_prices_v6_final_report.pdf) |
| 当前提交文件 | [`submission_v6.csv`](../submission_v6.csv) |
| 最终配方、权重、来源哈希 | [`experiments/v6/selection.json`](../experiments/v6/selection.json) |
| 模型重载和逐行预测检查 | [`experiments/v6/reproduction_audit.json`](../experiments/v6/reproduction_audit.json) |
| 版本历史 | [`VERSIONS.md`](../VERSIONS.md)、[`versions/index.json`](../versions/index.json) |
| 原始数据说明 | [`data/raw/data_description.txt`](../data/raw/data_description.txt) |

训练集有 **1,460 行**，测试集有 **1,459 行**。输入为 **79 个房屋特征**，另有 `Id`；训练集额外包含目标 `SalePrice`。`Id` 不参与建模，目标使用自然对数房价。官方 CSV 和原始下载 ZIP 都在代码包中，数据来源及校验值见 [`data/manifest.json`](../data/manifest.json)。

`selection.json` 保留模型定稿时的记录，里面的线上成绩仍为空；最终 **0.11212** 单独保存在 `experiments/v6/online_result.json` 和版本索引中，避免改写训练时的快照。

## 当前模型怎么组成

v6 将以下三个预测在**对数房价空间各取 1/3**，最后取指数恢复房价：

1. 已冻结的 v5 预测，包含传统模型集成和 SVR。
2. TabPFN v2，使用面积、浴室、房龄、质量交互等结构特征，内部集成成员数为 16。
3. TabPFN v2，使用原始特征，额外将 `OverallQual`、`OverallCond` 转为类别表示，内部集成成员数为 4。

模型包版本为 `tabpfn==9.2.0`，使用的预训练权重是 **TabPFN v2**。两者的版本名称含义不同。公开权重固定到具体 revision 和 SHA-256，已包含在仓库中；复现时不需要再次登录 Hugging Face 下载权重。

**Built with PriorLabs-TabPFN.** 权重许可及来源分别见 [`experiments/v6/pretrained/LICENSE.txt`](../experiments/v6/pretrained/LICENSE.txt) 和 [`manifest.json`](../experiments/v6/pretrained/manifest.json)。分享时请保留这两个文件及署名。

## 下载私有仓库或使用 ZIP

仓库地址：<https://github.com/WANGDAKEKE/kaggle-houseprice>。

这是私有仓库。仓库所有者需要邀请接收者成为 collaborator；接收者接受邀请后，才可以用自己的 GitHub 账号查看或克隆。只有仓库链接不会自动获得访问权限。邀请和仓库访问设置由所有者处理，本项目代码不会修改这些权限。

操作路径是仓库 **Settings → Collaborators → Add people**，填写接收者的 GitHub 用户名并邀请；对方接受后再使用仓库或 Release 链接。见 [GitHub 官方邀请步骤](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)。个人私有仓库的 collaborator 包含写权限；仅转发代码和报告时也可使用 ZIP。

有访问权限后，可以克隆固定分享版本：

```sh
git clone --branch v6.1.0 --depth 1 https://github.com/WANGDAKEKE/kaggle-houseprice.git
cd kaggle-houseprice
```

也可以直接解压本次提供的 ZIP，并进入包含 `src/`、`scripts/`、`data/` 的项目根目录。ZIP 是已提交文件的快照，不包含 Git 历史；希望继续协作、比较修改时优先克隆仓库。

本次分享的代码包文件名为 `kaggle-houseprice-v6.1.0.zip`，本机保存位置是 `delivery/kaggle-houseprice-v6.1.0.zip`。发布完成后，代码 ZIP、PDF 和单文件 HTML 会作为私有仓库 **v6.1.0 Release** 的附件提供；接收者获得仓库权限后，也可以从该 Release 下载。代码 ZIP 不会把自身再嵌套打包进去。

## 在原平台复现最新预测

已验证的平台是 **macOS Apple Silicon、Python 3.12.14、PyTorch MPS**。依赖版本保存在独立的 [`requirements-tabpfn.lock.txt`](../requirements-tabpfn.lock.txt)，早期传统模型使用另一个 [`requirements.lock.txt`](../requirements.lock.txt)。建议为两个环境分别建虚拟环境。

在 macOS 的项目根目录运行：

```sh
python3.12 -m venv .venv-tabpfn
.venv-tabpfn/bin/python -m pip install -r requirements-tabpfn.lock.txt
.venv-tabpfn/bin/python scripts/setup_openmp.py
sh scripts/run_tabpfn.sh scripts/predict_v6.py
```

`setup_openmp.py` 下载带固定校验值的 macOS OpenMP 动态库，放入项目自己的 `.runtime/`，供相关依赖导入使用。虚拟环境和这个动态库不打入分享包，由接收方在本机建立。

复现入口会读取保存的完整训练上下文和公开权重，重新计算两个 TabPFN 方向，然后与冻结的 `submission_v5.csv` 按三路等权融合；它不会重新搜索全部历史模型。输出包括：

- `experiments/v6/reproduced_submission_v6.csv`：重新计算的预测。
- `experiments/v6/reproduction_audit.json`：与已保存提交文件的最大对数预测差及历史文件校验。

运行前会核对数据、源码、锁文件、上下文及提交文件的 SHA-256。因此，**请保留原始锁文件和相关源码的字节内容**，先用干净快照完成重载检查，再另开版本修改模型。原机独立重载后的最大对数预测差约为 `1.59e-8`；脚本允许 `5e-5` 的数值误差，不要求 GPU 浮点预测逐位相同。

仅查看报告、比较已有 OOF 或使用已生成 CSV，不需要训练模型，也不需要安装全部机器学习依赖。

## Linux、Windows 或其他 Mac 的环境差异

TabPFN 安装包的官方说明及源码支持 CPU，并将 MLX 依赖限定在 Darwin/ARM64。当前分享的完整依赖锁是原 Mac 的安装记录，里面包含无平台条件的 `mlx`、`mlx-metal`；**不能直接当作 Linux/Windows 的通用依赖锁**。不同 macOS 版本也可能遇到某个固定版本缺少适配 wheel 的情况。

仓库提供了 [`requirements-tabpfn-portable.txt`](../requirements-tabpfn-portable.txt)，它仅从原锁文件移除了 `mlx` 和 `mlx-metal` 两行，供非 Mac ARM 平台安装时参考。原始锁文件保持不变，满足已有哈希核验；这个派生清单不代表已经验证跨平台可安装或重现。

在独立 Python 3.12 环境中安装派生清单；PyTorch 使用适合本机的官方构建。Linux 在环境中执行 `python scripts/predict_v6.py`；Windows 直接使用环境的 Python，例如：

```powershell
.\.venv-tabpfn\Scripts\python.exe -m pip install -r requirements-tabpfn-portable.txt
.\.venv-tabpfn\Scripts\python.exe scripts/predict_v6.py
```

Windows 需要先用本机 Python 建好 `.venv-tabpfn`；不要执行 macOS 的 OpenMP 安装脚本或 POSIX `sh` 包装器。

本项目当前设备选择逻辑是“有 MPS 用 MPS，否则用 CPU”，**不会自动使用 NVIDIA CUDA**。CPU 路径已在代码中允许超过 TabPFN v2 默认的样本数限制，但可能明显慢于原机 GPU。这里没有实际执行 Linux/Windows 全量重载，也没有验证这些系统上的固定依赖 wheel 和 `5e-5` 容差；跨平台数值差异应单独检查、记录。如果修改依赖或设备配置进行新实验，请另存结果，保留原提交及其审计。

## 包内有哪些复现材料

完整 Git 快照应包含：

- `src/`、`scripts/`：从早期模型到 v6 的训练、预测和报告代码。
- 两份原环境依赖锁及 portable 安装参考清单；`data/raw/` 官方数据、数据说明、来源清单及根目录官方 ZIP。
- `submission.csv`、`submission_v2.csv` 至 `submission_v6.csv`：已冻结的历史预测。
- `experiments/v6/fit_context.joblib`：两个 TabPFN 组件的训练数据、目标和精确参数。
- `experiments/v6/pretrained/`：公开 checkpoint、原许可和来源 manifest。
- `experiments/v6/selection.json`、提交审计、重载审计、测试组件预测、OOF、分组与逐房屋分析。
- 早期模型文件、缓存参考预测、版本快照，以及 v6 的 physical/half 备用方案，方便追溯比较。

最新预测重载直接使用冻结的 v5 CSV。若要继续追溯生成 v5 的传统模型，仓库同时保留对应模型文件、manifest、源码及 `scripts/predict_v5.py`；这一步需要早期模型环境，不是重建最新 v6 文件的必经步骤。

分享包不携带 `.git/`、`.venv/`、`.venv-tabpfn/`、`.runtime/`、`.cache/` 或登录凭据。用 `git archive` 打包固定标签，只会包含已提交的项目文件；运行环境和账号登录由接收方自行配置。

包清单检查记录见 [`docs/sharing_package_audit.json`](sharing_package_audit.json)。该记录区分了检查时的提交和待发布分享标签；最终发布还会核对实际 ZIP 内容及校验值。

打包脚本为 `python scripts/package_v6_share.py --tag v6.1.0`，需要本机 Git。它会从固定标签导出并核对 ZIP CRC、文件清单和关键模型哈希，生成 `delivery/share_manifest.json`；该文件随 Release 附件提供，含最终提交号与 ZIP 的 SHA-256。

## 理解本地成绩

当前 OOF 使用全部 1,460 条训练房屋的匹配五折 seed5509，验证房屋均保留。`ordinary` 只是在诊断表中排除两套异常低价大房屋，`Normal` 是普通成交类型；这些分组不改变主要评分的样本集合。

项目已用同一批历史房屋多轮比较模型和表示，最终的三路权重也参考了这些比较，存在方案选择偏差。本地分数适合解释不同方案的差异，不能当作全新的独立验证或预先保证的 Kaggle 分数。实际提交成绩以用户反馈和线上记录为准。
