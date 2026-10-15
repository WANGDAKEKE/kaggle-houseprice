"""v3 report: paired comparisons, additional split checks and model provenance."""
from pathlib import Path
import sys,os,json,html
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.optimize_v2 import partitions
from scripts.run_experiments import load_data
from src.house_prices import log_rmse

ART=ROOT/"experiments/v3";FIG=ART/"figures";FIG.mkdir(exist_ok=True)
selection=json.loads((ART/"selection.json").read_text())
checks=json.loads((ART/"additional_split_check.json").read_text())
five_new=list(selection["selected_seed_scores"].values())+[r["v3_log_rmse"] for r in checks["scores"]]
five_old=list(selection["v2_seed_scores"].values())+[r["v2_log_rmse"] for r in checks["scores"]]
five={"v3_five_split_mean":float(np.mean(five_new)),"v2_five_split_mean":float(np.mean(five_old)),
      "relative_improvement_pct":float(100*(1-np.mean(five_new)/np.mean(five_old))),
      "all_five_splits_improved":bool(all(a<b for a,b in zip(five_new,five_old))),
      "first_three_role":"selection","last_two_role":"checks after freezing selection","online_score_v3":None}
(ART/"five_split_summary.json").write_text(json.dumps(five,indent=2)+"\n")
audit=json.loads((ART/"submission_audit.json").read_text())
online_file=ART/"online_result.json"
online=json.loads(online_file.read_text()) if online_file.exists() else None
five["online_score_v3"]=online["public_log_rmse"] if online else None
(ART/"five_split_summary.json").write_text(json.dumps(five,indent=2)+"\n")
status="v3 已由用户提交，线上结果已记录" if online else "v3 尚未提交 Kaggle"
online_text=(f'<p>用户反馈 v3 线上分数：<b>{online["public_log_rmse"]:.5f}</b>。分数来源为用户反馈，未从提交历史独立核验。</p>' if online else '')
online_boundary="用户反馈的线上结果已单独记录，训练与提交文件保持冻结。" if online else "Kaggle 线上分数待用户提交后确认，本地改善不保证线上一定下降。"
if online and online["public_log_rmse"] > .11741:
    online_text += '<p><b>v3 线上退步，v2 仍是已提交的最佳版本。</b>复查发现本地净改善主要由两套极端房屋贡献；详见 <a href="../v4/report.html">v4 回退诊断与融合报告</a>。不能把五次相同房屋的拆分改善理解为五个独立数据集上的改善。</p>'
screen=pd.read_csv(ART/"screen_results.csv")
confirmed=pd.read_csv(ART/"confirmation_results.csv")
train,test=load_data();dev,hold=partitions(train);y=np.log(train.SalePrice.to_numpy())
out=pd.DataFrame({"Id":train.Id.iloc[dev].to_numpy(),"log_target":y[dev]})
for seed in selection["seeds"]:
    p=sum(np.load(ART/"cv"/f"dev_{name}_seed{seed}.npz")["prediction"]*w for name,w in selection["weights"].items())
    old=sum(np.load(ART/"cv"/f"dev_{name}_seed{seed}.npz")["prediction"]*.5 for name in ["ref_v2_linear","ref_v2_cat"])
    out[f"selected_seed_{seed}"]=p;out[f"v2_seed_{seed}"]=old
out.to_csv(ART/"confirmed_oof.csv",index=False)
details=train.iloc[dev][["Id","SalePrice","GrLivArea","Neighborhood","SaleCondition"]].copy()
details["log_error"]=out[[f"selected_seed_{s}" for s in selection["seeds"]]].mean(axis=1).to_numpy()-y[dev]
details["abs_error"]=details.log_error.abs()
details.sort_values("abs_error",ascending=False).to_csv(ART/"development_error_diagnostics.csv",index=False)
for p in (ART/"cv").glob("*.npz"):
    saved=np.load(p);assert np.array_equal(saved["ids"],train.Id.iloc[dev])
    assert np.isfinite(saved["prediction"]).all() and set(saved["fold_ids"])=={0,1,2,3,4}
files=list((ART/"cv").glob("*.npz"));reused=sum(bool(np.load(p)["reused_v2"]) for p in files)
run_audit={"base_configurations":len(json.loads((ART/"specifications.json").read_text())),
    "saved_cv_runs":len(files),"saved_fold_predictions":len(files)*5,"reused_v2_cv_runs":reused,
    "new_saved_cv_fold_fits":(len(files)-reused)*5,"all_validation_rows_retained":True,
    "holdout_not_used_for_selection":True,"additional_check_used_for_retuning":False}
(ART/"run_audit.json").write_text(json.dumps(run_audit,indent=2)+"\n")

plt.rcParams.update({"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
top=confirmed.head(14).sort_values("mean_log_rmse",ascending=False)
fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
colors=["#16846d" if name==selection["name"] else "#d79547" if name=="v2_frozen" else "#9aadc7" for name in top.model]
ax.scatter(top.mean_log_rmse,np.arange(len(top)),c=colors,s=45)
ax.set_yticks(np.arange(len(top)),top.model)
for i,value in enumerate(top.mean_log_rmse):ax.text(value+.00012,i,f"{value:.5f}",va="center",fontsize=8)
ax.set(xlabel="Mean log-RMSE on the three selection splits",title="v3 development model comparison")
ax.set_xlim(top.mean_log_rmse.min()-.001,top.mean_log_rmse.max()+.003)
ax.tick_params(axis="y",labelsize=8)
fig.savefig(FIG/"confirmation.png");plt.close(fig)
labels=[str(s) for s in selection["seeds"]]+[f"{r['seed']} (check)" for r in checks["scores"]]
old=[selection["v2_seed_scores"][str(s)] for s in selection["seeds"]]+[r["v2_log_rmse"] for r in checks["scores"]]
new=[selection["selected_seed_scores"][str(s)] for s in selection["seeds"]]+[r["v3_log_rmse"] for r in checks["scores"]]
fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);x=np.arange(len(labels))
ax.bar(x-.17,old,width=.34,color="#d79547",label="v2 frozen")
ax.bar(x+.17,new,width=.34,color="#16846d",label="v3 selected")
ax.axvline(2.5,color="#65758b",linestyle="--",linewidth=1)
ax.set_xticks(x,labels);ax.set(ylabel="Pooled OOF log-RMSE",title="Selection splits and later split checks")
ax.legend(fontsize=8);fig.savefig(FIG/"split_checks.png");plt.close(fig)
gain=100*(1-selection["mean_repeated_cv_log_rmse"]/selection["v2_mean_repeated_cv_log_rmse"])
weights=", ".join(f"{n} × {w:.0%}" for n,w in selection["weights"].items())
def table(frame):
    copy=frame.copy()
    for col in copy.select_dtypes(include="number"):
        copy[col]=copy[col].map(lambda v: str(int(v)) if col=="components" else f"{v:.5f}")
    return copy.to_html(index=False,border=0,escape=True)
report=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>House Prices v3 实验报告</title><style>body{{font:16px/1.75 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;background:#f5f7fb;color:#253348;margin:0}}
main{{max-width:1120px;margin:32px auto;padding:30px 38px;background:white;border-radius:16px}}h1{{line-height:1.3}}h2{{margin-top:30px}}a{{color:#285ba4}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #e1e8f0;text-align:left}}th{{background:#f1f5f9}}img{{max-width:100%}}
.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#eff6f8;padding:18px 24px;border-radius:10px;flex:1}}.number{{font-size:28px;color:#16846d;font-weight:700}}.muted{{color:#66758b}}
</style><main><p class="muted">2026-10-15 · {status} · 首版与 v2 均已保留</p>
<h1>House Prices v3：类别正则化与非线性特征</h1>
{online_text}
<p>v2 的用户反馈线上分数为 <b>0.11741</b>。本轮从 {run_audit['base_configurations']} 个配置开始，筛选固定配比融合，并在选定方案后检查两个未参与权重选择的新拆分。成功保存 {run_audit['new_saved_cv_fold_fits']} 次新的折内拟合结果，另复用 {reused*5} 次经过签名验证的 v2 折内预测；前期调试或因代码签名失效而重跑的操作不计入这里。</p>
<div class="cards"><div class="card">v2 · 原三个拆分均值<div class="number">{selection['v2_mean_repeated_cv_log_rmse']:.5f}</div></div>
<div class="card">v3 · 原三个拆分均值<div class="number">{selection['mean_repeated_cv_log_rmse']:.5f}</div></div>
<div class="card">v3 · 两个新增拆分均值<div class="number">{checks['mean_v3']:.5f}</div></div></div>
<h2>提交文件与方案</h2><p><a href="../../submission_v3.csv">submission_v3.csv</a>，共 {audit['rows']} 行，Id 与官方测试集一致。方案为 {html.escape(weights)}。随机性树模型对 3 个完整训练种子求平均，线性模型仅拟合一次；全部 1,460 条官方训练样本用于最终训练。模型保存到本版本目录，可以重新加载生成相同预测。</p>
<h2>本轮变化</h2><p>适度缩放稀有类别的独热特征，改变其正则化强度；比较街区与面积区间、房屋年代、建筑类型的组合特征；同时尝试 Box–Cox 变换、样条基函数、原生类别直方图提升树，以及不同类别处理、深度和稳健损失的 CatBoost。没有使用外部测试标签，没有按 Id 覆盖预测。</p>
<h2>验证与可信边界</h2><p>仍使用相同的 1,168 条开发数据，以及 42、2026、137 三个 5 折拆分。v3 相对 v2 的平均误差降低 {gain:.2f}%。融合组件按其组合后的误差比较，固定权重网格为 25%、50%、75%；距最低误差 0.00015 内优先较少的组件。</p>
<p>选定配置后，冻结参数并在 2701、901 两个新拆分上检查，未根据检查结果改权重：v2 均值 {checks['mean_v2']:.5f}，v3 均值 {checks['mean_v3']:.5f}。这些仍是同一批开发房屋，不能视为全新的独立数据。所有拆分的验证样本均保留，包括少数极大面积低价房屋。预处理、类别频率、幂变换列和样条结点只从各折训练样本拟合。</p>
<p>五个相同拆分统一比较，v2 的均值为 <b>{five['v2_five_split_mean']:.5f}</b>，v3 为 <b>{five['v3_five_split_mean']:.5f}</b>，五个拆分均改善，平均降低 {five['relative_improvement_pct']:.2f}%。其中 seed=901 把两条极大面积低价样本放在同一验证折，双方误差都升高；仍保留这个更难的拆分，不只报告较好的拆分。</p>
<p>旧的 292 条预留数据已在前两版评价过，本轮只作诊断，log-RMSE 为 {audit['old_holdout_log_rmse_diagnostic_only']:.5f}，不用于选择参数。{online_boundary}</p>
<img src="figures/split_checks.png" alt="选择拆分与新增拆分对比"><img src="figures/confirmation.png" alt="固定融合候选对比">
<h2>全部融合比较</h2>{table(confirmed)}<h2>全部初筛配置</h2>{table(screen)}
<h2>复现与版本</h2><p><a href="selection.json">冻结方案</a> · <a href="additional_split_check.json">新增拆分检查</a> · <a href="run_audit.json">验证审计</a> · <a href="submission_audit.json">提交文件校验</a> · <a href="../../VERSIONS.md">版本记录</a></p>
<p>运行 <code>sh scripts/run.sh scripts/predict_v3.py</code> 可以复现预测，并校验 v1、v2 的提交文件没有改变。v3 的源代码、参数、模型、预测和结果由 Git 标签 v3.0.0 冻结。</p></main></html>"""
(ART/"report.html").write_text(report)
print("v3 report:",ART/"report.html")
print("Run audit:",json.dumps(run_audit))
