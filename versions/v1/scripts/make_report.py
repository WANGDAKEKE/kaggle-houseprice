"""Data audit, static scientific figures and a readable local experiment report."""
from pathlib import Path
import sys
import os
import json
import html

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(ROOT / "data/raw/train.csv", keep_default_na=False, na_values=["NA"])
test = pd.read_csv(ROOT / "data/raw/test.csv", keep_default_na=False, na_values=["NA"])
results = pd.read_csv(ART / "experiments.csv")
selection = json.loads((ART / "selection.json").read_text())
holdout = json.loads((ART / "holdout_metrics.json").read_text())
validation = json.loads((ART / "submission_validation.json").read_text())
manifest = json.loads((ROOT / "data/manifest.json").read_text())
score_file = ART / "kaggle_results.json"
online_results = json.loads(score_file.read_text()).get("submissions", []) if score_file.exists() else []
online_html = ""
status_text = "本地实验已完成，尚未提交"
next_step_text = "本轮没有登录、上传、提交或读取榜单。下一轮提交推荐 CSV，记录线上 log-RMSE 后再决定后续实验。"
if online_results:
    latest = online_results[-1]
    status_text = "用户已提交首版，线上分数已记录"
    online_html = f'<h2>Kaggle 线上结果</h2><p>用户于 {html.escape(latest["date"])} 反馈首版 Public Score：<b>{latest["public_log_rmse"]:.5f}</b>，按对话记录关联推荐的 submission.csv。该分数由用户提供，未通过 Kaggle 接口独立核验。开发集 5 折验证是 {selection["selection_cv_log_rmse"]:.5f}，两者接近；样本不同，不能把差值解释为模型提升。</p>'
    next_step_text = "首版线上结果已保存到 kaggle_results.json。后续实验继续以本地交叉验证选择方案，再用有限的线上提交检查泛化表现。"
missing = pd.DataFrame({"train_missing": train.drop(columns="SalePrice").isna().sum(),
                        "test_missing": test.isna().sum()}).sort_values("train_missing", ascending=False)
missing.to_csv(ART / "missing_values.csv")
large_low_price = train.loc[(train.GrLivArea > 4000) & (train.SalePrice < 300000), ["Id", "GrLivArea", "SalePrice"]]
audit = {"train_shape": list(train.shape), "test_shape": list(test.shape), "feature_count": 79,
         "training_rows_removed": 0, "target": train.SalePrice.describe().to_dict(),
         "large_low_price_diagnostic_only": large_low_price.to_dict("records"),
         "duplicate_train_ids": int(train.Id.duplicated().sum()),
         "duplicate_test_ids": int(test.Id.duplicated().sum()),
         "future_garage_year_ids": train.loc[train.GarageYrBlt > train.YrSold, "Id"].tolist(),
         "masvnrtype_missing": int(train.MasVnrType.isna().sum()),
         "masvnrtype_none_count": int((train.MasVnrType == "None").sum())}
(ART / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
axes[0].hist(train.SalePrice / 1000, bins=35, color="#3564a5", alpha=0.9)
axes[0].set(title="Sale price distribution", xlabel="Price ($ thousands)", ylabel="Homes")
axes[1].hist(np.log(train.SalePrice), bins=35, color="#218570", alpha=0.9)
axes[1].set(title="Natural log of sale price", xlabel="ln(SalePrice)", ylabel="Homes")
fig.savefig(FIG / "target_distribution.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ordered = results.sort_values("oof_log_rmse", ascending=False)
colors = ["#218570" if name == selection["name"] else "#9aadc6" for name in ordered.model]
ax.barh(ordered.model, ordered.oof_log_rmse, color=colors)
ax.set(xlabel="5-fold pooled OOF log-RMSE (lower is better)", title="Model comparison: development data only")
ax.set_xlim(0.10, max(ordered.oof_log_rmse) + 0.012)
for i, value in enumerate(ordered.oof_log_rmse):
    ax.text(value + 0.0004, i, f"{value:.4f}", va="center", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
fig.savefig(FIG / "model_comparison.png")
plt.close(fig)

oof = pd.read_csv(ART / "development_oof.csv")
dev_details = train.merge(oof[["Id", "log_target", selection["name"]]], on="Id", how="inner")
dev_details["log_error"] = dev_details[selection["name"]] - dev_details.log_target
dev_details["abs_log_error"] = dev_details.log_error.abs()
dev_details.sort_values("abs_log_error", ascending=False)[["Id", "SalePrice", "GrLivArea", "OverallQual", "log_error"]].to_csv(ART / "largest_development_errors.csv", index=False)
normal_mask = ~((dev_details.GrLivArea > 4000) & (dev_details.SalePrice < 300000))
diagnostic = {"all_development_log_rmse": float(np.sqrt(np.mean(dev_details.log_error ** 2))),
              "excluding_two_large_low_price_homes_diagnostic_only": float(np.sqrt(np.mean(dev_details.loc[normal_mask, "log_error"] ** 2))),
              "excluded_count_for_diagnostic": int((~normal_mask).sum()),
              "note": "Diagnostic only; all official rows were retained in training and primary evaluation."}
(ART / "error_diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
hp = pd.read_csv(ART / "holdout_predictions.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].scatter(hp.log_target, hp.log_prediction, s=17, alpha=0.6, color="#3564a5")
lims = [min(hp.log_target.min(), hp.log_prediction.min()) - .1, max(hp.log_target.max(), hp.log_prediction.max()) + .1]
axes[0].plot(lims, lims, color="#7b8794", linestyle="--")
axes[0].set(title="Locked holdout predictions", xlabel="Actual ln(price)", ylabel="Predicted ln(price)")
axes[1].scatter(hp.log_prediction, hp.log_error, s=17, alpha=0.6, color="#218570")
axes[1].axhline(0, color="#7b8794", linestyle="--")
axes[1].set(title="Locked holdout residuals", xlabel="Predicted ln(price)", ylabel="Prediction minus actual")
fig.savefig(FIG / "holdout_diagnostics.png")
plt.close(fig)

baseline_cv = float(results.loc[results.model == "ridge_baseline", "oof_log_rmse"].iloc[0])
improvement = 100 * (1 - selection["selection_cv_log_rmse"] / baseline_cv)
hold_improvement = 100 * (1 - holdout["selected_log_rmse"] / holdout["baseline_log_rmse"])
weights_text = ", ".join(f"{name} × {weight:.0%}" for name, weight in selection["weights"].items())
table = results[["model", "oof_log_rmse", "fold_std", "seconds"]].copy()
for col in ["oof_log_rmse", "fold_std"]:
    table[col] = table[col].map(lambda x: f"{x:.5f}")
table["seconds"] = table.seconds.map(lambda x: f"{x:.1f}")
table.columns = ["模型", "OOF log-RMSE", "折间标准差", "5 折总耗时（秒）"]
report = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Prices 实验报告</title><style>
body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#233347;background:#f5f7fb;margin:0}}
main{{max-width:1050px;margin:36px auto;padding:32px 40px;background:white;border-radius:18px}}
h1{{line-height:1.25}} h2{{margin-top:32px}} p{{max-width:940px}} .muted{{color:#64748b}}
.cards{{display:flex;gap:18px;flex-wrap:wrap}} .card{{background:#eef5f8;padding:18px 24px;border-radius:12px;flex:1;min-width:180px}}
.number{{font-size:30px;font-weight:700;color:#146b5b}} img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{border-bottom:1px solid #e2e8f0;padding:8px 12px;text-align:left}}
th{{background:#f1f5f9}}code{{background:#f1f5f9;padding:2px 6px;border-radius:4px}}a{{color:#285ba4}}
</style><main>
<p class="muted">Kaggle · House Prices · 2026-10-15 · {status_text}</p>
<h1>房价预测：基线、特征工程与模型融合</h1>
<p>使用你提供的官方 ZIP。训练集 1,460 行、79 个特征；测试集 1,459 行。目标采用自然对数 <code>ln(SalePrice)</code>，预测后通过 <code>exp</code> 转回美元。log-RMSE 越小越好。</p>
<div class="cards"><div class="card">Ridge 基线 · 开发集 5 折<div class="number">{baseline_cv:.5f}</div></div>
<div class="card">最终方案 · 开发集 5 折<div class="number">{selection['selection_cv_log_rmse']:.5f}</div></div>
<div class="card">最终方案 · 预留检查集<div class="number">{holdout['selected_log_rmse']:.5f}</div></div></div>
{online_html}
<h2>最终选择</h2><p><b>{html.escape(selection['name'])}</b>：{html.escape(weights_text)}。在对数空间加权，因此最终房价是各模型价格预测的加权几何平均。开发集相对 Ridge 基线降低 {improvement:.1f}% 的 log-RMSE。绝对最低的候选分数是 {selection['best_candidate_cv_log_rmse']:.5f}；在距最低值 0.0003 以内的方案中，优先选择模型数量更少的组合，再比较误差。此选择在查看预留集之前完成。</p>
<p>已生成 <a href="../submission.csv">submission.csv</a>（推荐）以及 <a href="../submission_baseline.csv">submission_baseline.csv</a>（对照）。检查了行数、列名、Id 顺序、有限且为正的预测值。推荐文件价格范围：${validation['min_price']:,.0f}–${validation['max_price']:,.0f}。</p>
<h2>验证设计与可信边界</h2>
<p>按房价十分位分层，固定 seed=42 留出 292 行（20%）作为最终检查集；其余 1,168 行用于 5 折交叉验证。缺失值填补、偏态列选择、缩放和类别编码均只在各折训练数据上拟合。Id 不参与建模。验证集从未用于早停，也没有删除异常验证样本。</p>
<p>先比较 {len(results)} 个候选方案（含固定权重融合），仅按开发集 OOF 指标选定模型，再一次性检查预留数据；检查后不改模型或参数。预留集 Ridge 基线为 {holdout['baseline_log_rmse']:.5f}，所选方案为 {holdout['selected_log_rmse']:.5f}，变化为 {hold_improvement:.1f}% 改善。开发集最优分数有模型选择偏差，预留结果只代表这个拆分；两者都不是 Kaggle 线上分数。</p>
<p>最后用全部 1,460 行重新训练选定模型，保留拟合后的模型及预测分量。测试集没有真实房价，不能在本地算出比赛分数。</p>
<h2>数据与特征处理</h2>
<p>区分字符串 <code>None</code> 和真正的 <code>NA</code>：MasVnrType 的 {audit['masvnrtype_none_count']} 个 None 表示无石材贴面，真正缺失只有 {audit['masvnrtype_missing']} 条。数值缺失采用训练折中位数，类别缺失使用独立标记；MSSubClass 按类别处理。</p>
<p>加入总面积、完成面积、卫浴数量、户外面积、房龄、装修后年数、设施有无、质量等级、质量与面积交互、月度周期等特征。线性模型对训练折中偏态的非负数值做 log1p 变换。CatBoost 使用原生类别特征。</p>
<img src="figures/target_distribution.png" alt="房价及其对数分布">
<h2>所有实验</h2><img src="figures/model_comparison.png" alt="开发集模型对比">{table.to_html(index=False, border=0, escape=True)}
<h2>误差诊断</h2><p>Id 524 和 1299 的居住面积超过 4,000 平方英尺，但售价低于 $300,000；它们会显著影响部分折的误差。本轮保留全部样本。仅供诊断：开发集排除其中的 {diagnostic['excluded_count_for_diagnostic']} 条后的 log-RMSE 是 {diagnostic['excluding_two_large_low_price_homes_diagnostic_only']:.5f}，这个数不参与模型选择，也不应与完整样本分数混用。</p>
<img src="figures/holdout_diagnostics.png" alt="预留集预测与误差诊断">
<h2>文件与复现</h2><p><a href="experiments.csv">实验指标 CSV</a> · <a href="selection.json">固定方案与融合权重</a> · <a href="holdout_metrics.json">预留集指标</a> · <a href="../data/manifest.json">官方数据校验记录</a> · <a href="../README.md">运行说明</a></p>
<p class="muted">{next_step_text} <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques">比赛页面</a></p>
</main></html>"""
(ART / "report.html").write_text(report)
print("Report:", ART / "report.html")
print("Data audit:", json.dumps(audit))
