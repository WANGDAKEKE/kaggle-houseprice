"""Generate the versioned experiment report after freezing the v2 selection."""
from pathlib import Path
import os
import json
import html

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = ROOT / "experiments/v2"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
screen = pd.read_csv(ART / "screen_results.csv")
confirmed = pd.read_csv(ART / "confirmation_results.csv")
selection = json.loads((ART / "selection.json").read_text())
hold = json.loads((ART / "holdout_diagnostic.json").read_text())
audit = json.loads((ART / "submission_audit.json").read_text())
comparison = pd.read_csv(ART / "submission_comparison.csv")
old_hold = json.loads((ROOT / "versions/v1/artifacts/holdout_metrics.json").read_text())["selected_log_rmse"]
improvement = selection["v1_mean_repeated_cv_log_rmse"] - selection["mean_repeated_cv_log_rmse"]
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
top = confirmed.head(15).sort_values("mean_seed_rmse", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
colors = ["#1b806a" if name == selection["name"] else "#ce8b45" if name == "v1_fixed_blend" else "#9aadc7" for name in top.model]
ax.barh(top.model, top.mean_seed_rmse, color=colors)
for i, value in enumerate(top.mean_seed_rmse):
    ax.text(value + .0002, i, f"{value:.5f}", va="center", fontsize=8)
ax.set_xlim(min(top.mean_seed_rmse) - .005, max(top.mean_seed_rmse) + .006)
ax.set(xlabel="Mean log-RMSE over three 5-fold development splits", title="v2 repeated-CV confirmation")
ax.tick_params(axis="y", labelsize=9)
fig.savefig(FIG / "confirmation.png")
plt.close(fig)
seeds = selection["seeds"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
xs = np.arange(len(seeds))
axes[0].bar(xs - .17, [selection["v1_seed_scores"][str(s)] for s in seeds], width=.34, color="#ce8b45", label="v1 fixed blend")
axes[0].bar(xs + .17, [selection["selected_seed_scores"][str(s)] for s in seeds], width=.34, color="#1b806a", label="v2 selected")
axes[0].set_xticks(xs, [str(s) for s in seeds])
axes[0].set(ylabel="Pooled OOF log-RMSE", xlabel="Split seed", title="Same houses, paired CV comparison")
axes[0].legend(fontsize=8)
axes[1].hist(comparison.ratio_v2_v1, bins=40, color="#3564a5")
axes[1].axvline(1, linestyle="--", color="#6a7786")
axes[1].set(xlabel="v2 / v1 predicted price", ylabel="Test homes", title="Prediction changes")
fig.savefig(FIG / "version_comparison.png")
plt.close(fig)

def display_table(frame):
    f = frame.copy()
    for col in f.select_dtypes(include="number"):
        f[col] = f[col].map(lambda x: f"{x:.5f}" if pd.notna(x) else "—")
    return f.to_html(index=False, border=0, escape=True)

weights = ", ".join(f"{name} × {w:.0%}" for name, w in selection["weights"].items())
base_count = len(json.loads((ART / "specifications.json").read_text()))
fits = len(list((ART / "cv").glob("*.npz"))) * 5
ci = selection["paired_bootstrap_improvement_95pct"]
report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>House Prices v2 实验</title><style>
body{{font:16px/1.75 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#233347;background:#f5f7fb;margin:0}}
main{{max-width:1080px;margin:36px auto;padding:30px 40px;background:white;border-radius:18px}}h1{{line-height:1.3}}h2{{margin-top:30px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{flex:1;min-width:200px;padding:18px 22px;border-radius:10px;background:#eff6f8}}.number{{font-size:30px;color:#1b806a;font-weight:700}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 9px;border-bottom:1px solid #e1e8f0;text-align:left}}th{{background:#f1f5f9}}a{{color:#285ba4}}.muted{{color:#64748b}}code{{background:#f1f5f9;padding:2px 5px}}
</style><main><p class="muted">2026-10-15 · 私有 GitHub 版本管理 · v2 尚未提交 Kaggle</p>
<h1>House Prices v2：更细特征与重复验证</h1>
<p>首版线上分数为用户反馈的 <b>0.11969</b>。本轮筛选 {base_count} 个单模型配置，在 3 个随机种子的 5 折拆分上复核候选，再比较固定权重融合；共完成 {fits} 次折内训练。</p>
<div class="cards"><div class="card">v1 · 3 次 5 折均值<div class="number">{selection['v1_mean_repeated_cv_log_rmse']:.5f}</div></div>
<div class="card">v2 · 3 次 5 折均值<div class="number">{selection['mean_repeated_cv_log_rmse']:.5f}</div></div>
<div class="card">v2 旧预留集 · 仅诊断<div class="number">{hold['log_rmse']:.5f}</div></div></div>
<h2>新提交文件</h2><p>推荐 <a href="../../submission_v2.csv">submission_v2.csv</a>，方案为 <b>{html.escape(selection['name'])}</b>：{html.escape(weights)}。随机性树模型使用 3 个训练种子的全量模型平均，确定性模型只拟合一次，全部训练样本参与训练。融合在对数空间完成。</p>
<p>新文件 {audit['rows']} 行，Id 顺序与官方测试集一致，价格均有限且为正。预测范围 ${audit['min_price']:,.0f}–${audit['max_price']:,.0f}。与 v1 相比，价格比值中位数 {audit['median_prediction_ratio_to_v1']:.4f}，1%–99% 分位为 {audit['p01_prediction_ratio_to_v1']:.4f}–{audit['p99_prediction_ratio_to_v1']:.4f}。</p>
<h2>评价方法与边界</h2><p>继续使用首版的 1,168 条开发样本；各次拆分均保留所有验证样本，5 折种子为 42、2026、137。按三个拆分的平均 OOF log-RMSE 选择；距最低值 0.0002 内优先较少组件。首版冻结方案在完全相同拆分上重新计算，确保比较公平。缺失值、邻域中位数、裁剪阈值、偏态列、缩放和类别编码均只从每折训练数据拟合。</p>
<p>v2 相对 v1 的平均 CV 变化为 {improvement:.5f}（正数代表改善）。开发集按房屋配对的探索性 bootstrap 改善区间为 [{ci[0]:.5f}, {ci[1]:.5f}]；候选和权重已经过模型选择，这个区间不是独立统计保证。</p>
<p>v1 曾评价的 292 条预留数据已经不是未见过的检查集。本轮选定配置后才计算其误差，仅作诊断，未用它选择方案或继续调参：v1={old_hold:.5f}，v2={hold['log_rmse']:.5f}。本地分数不能保证新文件的 Kaggle 分数更低，需由用户提交后验证。</p>
<h2>本轮尝试</h2><p>按字段说明区分设施不存在与真正未知；LotFrontage 使用训练折内 Neighborhood 中位数。增加地下室完成比例、车位面积、质量与面积交互、土地面积比例、房屋质量组合等特征。比较 ElasticNet、Lasso、Ridge、SVR、KernelRidge、CatBoost、XGBoost 和 LightGBM。</p>
<p>另比较训练折内面积分位裁剪和两条大面积低价样本的训练权重降低。所有主指标始终包含这两条验证样本；screen_results.csv 中普通样本诊断列不参与最终选择。</p>
<img src="figures/confirmation.png" alt="重复验证模型对比"><img src="figures/version_comparison.png" alt="各随机拆分与预测变化">
<h2>重复验证全部方案</h2>{display_table(confirmed)}
<h2>初筛全部配置</h2>{display_table(screen)}
<h2>复现与版本</h2><p><a href="selection.json">固定方案</a> · <a href="model_manifest.json">最终模型与种子</a> · <a href="confirmation_results.csv">重复验证结果</a> · <a href="submission_audit.json">新文件校验</a> · <a href="../../VERSIONS.md">版本记录</a> · <a href="https://github.com/WANGDAKEKE/kaggle-houseprice">GitHub 私有仓库</a></p>
<p>运行 <code>sh scripts/run.sh scripts/predict_v2.py</code>，可从已保存模型复现新文件，并检查 v1 原文件的 SHA-256 未变。首版源代码与结果冻结在 <code>v1.0.0</code> 标签；v2 使用新的标签与文件名。</p>
</main></html>"""
(ART / "report.html").write_text(report)
print("v2 report:", ART / "report.html")
print("Training fits:", fits)
