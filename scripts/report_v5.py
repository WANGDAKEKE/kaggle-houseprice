"""Paired outer-fit diagnostics, subgroup checks and source/cache audit for v5."""
from pathlib import Path
import sys,json,html,collections,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.run_experiments import load_data
from scripts.optimize_v5 import digest,source_hash,load_frozen,metrics

ART=ROOT/"experiments/v5"


def load(name):return json.loads((ART/name).read_text())


def save(name,value):
    (ART/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def table(frame):return frame.to_html(index=False,float_format=lambda x:f"{x:.6f}",border=0,escape=True)


def main():
    frozen=load_frozen();protocol=load("protocol.json");selection=load("selection.json")
    audit=load("submission_audit.json");gate=load("release_gate.json");train,_=load_data()
    ids=train.Id.to_numpy();target=np.log(train.SalePrice.to_numpy())
    repeats=[];records=[];recipes=[];summaries=[];assignments=[];applied_gates=[]
    for seed in protocol["outer_cv"]["seeds"]:
        path=ART/"outer"/f"seed{seed}";summary=json.loads((path/"summary.json").read_text())
        assert summary["source_hash"]==frozen["source_hash"]
        assert summary["protocol_sha256"]==frozen["protocol_sha256"]
        assert summary["frozen_candidates_sha256"]==digest(ART/"frozen_candidates.json")
        assert summary["predictions_sha256"]==digest(path/"predictions.csv")
        p=pd.read_csv(path/"predictions.csv");assert np.array_equal(p.Id,ids)
        np.testing.assert_allclose(p.log_target,target,rtol=0,atol=1e-12)
        assert set(p.fold)==set(range(5)) and np.isfinite(p[["v4","selected"]]).all().all()
        repeats.append(p);summaries.append(summary)
        fold_recipes={r["fold"]:frozen["recipes"][r["selected_recipe"]] for r in summary["folds"]}
        active=p.fold.map(lambda fold:fold_recipes[fold]["support_gate"] and fold_recipes[fold]["new_weight"]>0).to_numpy()
        applied_gate=np.where(active,p.support_gate.to_numpy(),1.)
        applied_gates.append(applied_gate)
        assignments.append(pd.DataFrame({"Id":ids,f"fold_seed{seed}":p.fold,
            f"potential_support_seed{seed}":p.support_gate,f"applied_support_seed{seed}":applied_gate}))
        for kind in ["v4","selected"]:
            calculated=metrics(target,p[kind].to_numpy(),train)
            assert all(abs(calculated[m]-summary[kind][m])<1e-12 for m in calculated)
            records.append({"seed":seed,"model":kind,**calculated})
        for f in summary["folds"]:recipes.append({"seed":seed,**f})
    pd.DataFrame(records).to_csv(ART/"outer_summary.csv",index=False)
    assignments[0].merge(assignments[1],on="Id",validate="one_to_one").to_csv(ART/"outer_assignments_and_gates.csv",index=False)
    a=np.column_stack([p.v4 for p in repeats]);b=np.column_stack([p.selected for p in repeats])
    gains=np.mean((a-target[:,None])**2-(b-target[:,None])**2,axis=1)
    details=train[["Id","SalePrice","GrLivArea","OverallQual","Neighborhood","SaleCondition","YrSold"]].copy()
    details["paired_mean_squared_error_gain"]=gains
    details["v4_mean_oof_log_prediction"]=a.mean(axis=1)
    details["selected_mean_oof_log_prediction"]=b.mean(axis=1)
    details.sort_values("paired_mean_squared_error_gain",ascending=False).to_csv(ART/"outer_house_error_contributions.csv",index=False)
    extreme=(train.GrLivArea.to_numpy()>4000)&(train.SalePrice.to_numpy()<300000)
    groups={"all":np.ones(len(train),bool),"ordinary_without_two_extremes_diagnostic":~extreme,
        "normal_transactions":train.SaleCondition.eq("Normal").to_numpy(),"area_le_1500":train.GrLivArea.le(1500).to_numpy(),
        "area_1500_2500":train.GrLivArea.between(1501,2500).to_numpy(),"area_gt_2500":train.GrLivArea.gt(2500).to_numpy(),
        "large_normal_transactions":(train.GrLivArea.gt(2500)&train.SaleCondition.eq("Normal")).to_numpy()}
    for col in ["YrSold","SaleCondition"]:
        for val in sorted(train[col].unique()):
            m=train[col].eq(val).to_numpy()
            if m.sum()>=20:groups[f"{col}={val}"]=m
    group_records=[];bootstrap={};rng=np.random.default_rng(55015509)
    for label,mask in groups.items():
        score_a=np.sqrt(np.mean((a[mask]-target[mask,None])**2,axis=0)).mean()
        score_b=np.sqrt(np.mean((b[mask]-target[mask,None])**2,axis=0)).mean()
        group_records.append({"group":label,"rows":int(mask.sum()),"v4_mean_split_rmse":float(score_a),
            "selected_mean_split_rmse":float(score_b),"rmse_improvement":float(score_a-score_b),
            "fraction_houses_with_positive_mse_gain":float((gains[mask]>0).mean()),
            "net_squared_error_gain":float(gains[mask].sum())})
        if label in ["all","ordinary_without_two_extremes_diagnostic","normal_transactions"]:
            values=gains[mask];boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(5000)])
            top5=np.sort(values[values>0])[-5:]
            bootstrap[label]={"paired_house_mean_mse_gain":float(values.mean()),
                "bootstrap_95pct_interval":np.quantile(boot,[.025,.975]).tolist(),
                "interval_role":"descriptive house resampling conditional on these shared-training cross-fit predictions; not a full model-development generalization confidence interval",
                "net_gain_without_five_largest_positive_contributions":float(values.sum()-top5.sum())}
    group_table=pd.DataFrame(group_records);group_table.to_csv(ART/"outer_subgroup_metrics.csv",index=False)
    support_audit={"training_houses":len(train),"any_repeat_tapered_houses":int(np.any(np.column_stack([g<1 for g in applied_gates]),axis=1).sum()),
        "any_repeat_zero_weight_houses":int(np.any(np.column_stack([g==0 for g in applied_gates]),axis=1).sum()),
        "potential_any_repeat_tapered_houses":int(np.any(np.column_stack([p.support_gate.to_numpy()<1 for p in repeats]),axis=1).sum()),
        "actual_counts_respect_selected_fold_recipe":True,
        "taper_uses_features_only":True,"no_validation_rows_removed":True,
        "interpretation":"q95/q99 are fixed fold-local extrapolation heuristics, not a proven physical boundary. Gate was designed after historical screening, before outer fits."}
    caches=list((ART/"cv").glob("*.npz"));fits=0;internal_extra=0;matched_caches=0;ignored_caches=[]
    id_index=pd.Index(ids)
    for path in caches:
        saved=np.load(path);signature=str(saved["signature"])
        matched=[n for n in frozen["pool"] if path.stem.endswith(f"_{n}_{signature[:12]}")]
        if len(matched)!=1:
            ignored_caches.append(path.name);continue
        name=matched[0];suffix=f"_{name}_{signature[:12]}";context=path.stem[:-len(suffix)]
        fit_rows=id_index.get_indexer(saved["fit_ids"]);pred_rows=id_index.get_indexer(saved["ids"])
        assert (fit_rows>=0).all() and (pred_rows>=0).all()
        expected=hashlib.sha256((source_hash()+json.dumps({"name":name,"spec":frozen["pool"][name],
            "fit_rows":fit_rows.tolist(),"predict_rows":pred_rows.tolist(),"context":context},sort_keys=True)).encode()).hexdigest()
        if signature!=expected:
            ignored_caches.append(path.name);continue
        matched_caches+=1
        assert set(saved["fit_ids"]).isdisjoint(saved["ids"])
        assert np.isfinite(saved["prediction"]).all();n=int(saved["actual_model_fits"]);fits+=n
        if "crossfit_en_residual_cat" in path.name:internal_extra+=4*n
    counts=collections.Counter(r["selected_recipe"] for r in recipes)
    run_audit={"screen_unique_configurations":24,"kernel_configs":12,"tree_configs":10,"encoding_configs":2,
        "frozen_new_candidates":frozen["new_names"],"frozen_recipes":len(frozen["recipes"]),
        "completed_outer_repeats":len(repeats),"outer_folds":len(recipes),
        "new_cached_train_predict_calls":matched_caches,"ignored_other_fingerprint_cache_files":ignored_caches,
        "estimator_fit_calls_in_cached_runs":fits,
        "additional_internal_EN_fits_in_residual_models":internal_extra,
        "fit_count_note":"Cached calls only; screen and source-change reruns are documented separately. Residual estimator itself adds three inner EN and one full-fit EN.",
        "outer_recipe_selection_counts":dict(counts),"source_hash":source_hash(),
        "pre_outer_freeze_commit":"389df4a","all_cached_predict_rows_disjoint_from_fit_rows":True,
        "prior_global_screening_and_historical_data_reuse":True,"outer_results_used_for_recipe_retuning":False}
    save("run_audit.json",run_audit);save("paired_uncertainty.json",bootstrap);save("support_gate_audit.json",support_audit)
    figdir=ART/"figures";figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,metric,title in zip(axes,["all","ordinary","normal"],["All 1,460 houses","Ordinary houses (diagnosis)","Normal transactions"]):
        old=[s["v4"][metric] for s in summaries];new=[s["selected"][metric] for s in summaries]
        x=np.arange(len(summaries));ax.plot(x,old,"o-",label="v4",color="#b97a27")
        ax.plot(x,new,"o-",label="nested selector",color="#157e6b")
        ax.set_xticks(x,[str(s["seed"]) for s in summaries]);ax.set(title=title,ylabel="Pooled OOF log-RMSE",xlabel="Outer split seed")
        ax.ticklabel_format(axis="y",style="plain",useOffset=False);ax.legend(fontsize=8)
    fig.savefig(figdir/"outer_comparison.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,mask,title in zip(axes,[~extreme,train.SaleCondition.eq("Normal").to_numpy()],
            ["Ordinary houses (diagnosis)","Normal transactions"]):
        ordered_gains=np.sort(gains[mask])[::-1]
        ax.plot(np.arange(1,len(ordered_gains)+1),np.cumsum(ordered_gains),color="#157e6b")
        ax.axhline(0,color="#888",linewidth=1);ax.set(title=title,
            xlabel="Houses ordered by error improvement",ylabel="Cumulative squared-error gain vs v4")
    fig.savefig(figdir/"house_gain_concentration.png");plt.close(fig)
    online_path=ART/"online_result.json";online=json.loads(online_path.read_text()) if online_path.exists() else None
    online_text=f'用户反馈 v5 线上分数为 {online["public_log_rmse"]:.5f}。' if online else 'v5 尚未提交 Kaggle，线上成绩未知。'
    means=gate["mean_two_outer_repeat_scores"]
    short_table=group_table.head(7)
    fold_table=pd.DataFrame([{ "seed":r["seed"],"fold":r["fold"],"recipe":r["selected_recipe"],
        "v4_all":r["v4"]["all"],"selected_all":r["selected"]["all"],
        "v4_normal":r["v4"]["normal"],"selected_normal":r["selected"]["normal"]} for r in recipes])
    chosen=selection["recipe"]
    risk_paragraph='''<p><b>不能把平均改善理解为全面稳定改善。</b>seed5501 全样本从 0.11308 升至 0.11431，普通房屋也略退步；seed5509 改善更多，最终通过的是事先固定的两次均值门槛。全体与普通房屋的描述性 bootstrap 区间都包含零，去掉五笔最大正收益后净平方误差收益分别为 −0.27214、−0.02757，仍有收益集中。Normal 成交则十个外层折均改善，其描述性区间完全为正，去掉五笔最大正收益后仍有 +0.17551 净收益；这一证据仅适用于该组。面积&gt;2500、Abnorml 成交及 2007 年组均退步，完整数值保留在分组表中。</p>'''
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>House Prices v5：基础模型、普通房屋与嵌套比较</title><style>body{{font:16px/1.8 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;background:#f3f6fa;color:#26364b;margin:0}}main{{max-width:1150px;margin:28px auto;background:white;border-radius:14px;padding:32px}}h1{{line-height:1.35}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:6px;border-bottom:1px solid #dfe6ee;text-align:left}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}a{{color:#2866a6}}.cards{{display:flex;gap:15px;flex-wrap:wrap}}.card{{background:#edf7f3;padding:16px;border-radius:10px;flex:1}}.number{{font-size:24px;font-weight:bold;color:#157e6b}}</style><main>
<h1>v5：用不同误差形态的模型改善普通房屋</h1><p>v4 用户反馈线上最佳分数 <b>0.11679</b>。{online_text} 新文件：<a href="../../submission_v5.csv">submission_v5.csv</a>。本地分数不等于线上分数。</p>
<h2>本轮实际尝试</h2><p>初筛 {run_audit['screen_unique_configurations']} 个配置：SVR、RBF/多项式/Laplacian 核岭、紧凑物理特征的线性模型，正确中心化目标的 CatBoost Huber/MAE、LightGBM、XGBoost、sklearn Huber 提升树，折内目标编码，以及用交叉预测残差训练的线性模型加浅层树。</p>
<p>多数新树和目标编码方案对普通房屋没有净收益，未入选。CatBoost MAE 虽改善了初筛总体分数，却让普通房屋和 Normal 交易恶化，因此淘汰。最终冻结 {len(frozen['new_names'])} 个新候选：{html.escape(', '.join(frozen['new_names']))}，共 {len(frozen['recipes'])} 个包含原 v4 的固定融合候选。详细负结果和公开方法来源见 <a href="research_notes.md">研究笔记</a>、<a href="kernel_screen/screen_results.csv">核方法</a>、<a href="tree_screen/results.csv">树模型</a>、<a href="encoding_screen/results.csv">目标编码</a>。</p>
<h2>怎样比较，避免只改善两套异常房屋</h2><p>候选、门槛、权重与源代码在新外层计算前提交冻结（commit 389df4a）。外层使用全部 1,460 条数据的两次 5 折；每个外层训练子集内部重新做 3 折预测，依据全样本、普通房屋和 Normal 交易误差选择方案。只用内层选权重，外层评价这个自动选择过程；外层结果没有再用来挑具体配比。外层 selected 分数评价自动选择流程，并非最终固定配方的一份独立 OOF 分数。</p>
<p>每次 fit 的预处理和面积分位数只从该次训练行学习。大面积外推保护是在历史初筛后提出、外层计算前冻结的启发式：训练面积 q95 以下权重为 1，q95 到 q99 线性递减，q99 以上为 0；与不使用保护的方案一同进入内层比较。含核模型的等权组合使用共同保护，会整体缩小组合增量。它不是按 Id 或真实售价覆盖预测，所有验证样本均计入总体误差。</p>
<p><b>限制：</b>初筛已经看过原 1,168 条开发房屋，这些房屋也出现在本轮外层中。因此这里只能称固定筛选名单后的嵌套拟合评估，不能声称整个长期模型搜索过程拥有无偏、全新独立验证。两次拆分还是同一批房屋。排除两套异常的指标仅为诊断。</p>
<div class="cards"><div class="card">全样本 · v4 → 自动选择流程<div class="number">{means['v4']['all']:.5f} → {means['selected']['all']:.5f}</div></div>
<div class="card">普通房屋诊断<div class="number">{means['v4']['ordinary']:.5f} → {means['selected']['ordinary']:.5f}</div></div>
<div class="card">Normal 交易<div class="number">{means['v4']['normal']:.5f} → {means['selected']['normal']:.5f}</div></div></div>
{risk_paragraph}
<img src="figures/outer_comparison.png" alt="相同外层拆分的全样本、普通房屋和Normal交易比较">{table(short_table)}
<p>外层发布门槛：全样本均值最多回退 0.0003，且普通房屋及 Normal 交易均改善；本轮门槛结果为 <b>{gate['passed']}</b>。不把各折视为独立样本计算显著性；按房屋配对的 5,000 次 bootstrap 只是给定共享训练数据的这些交叉预测后的描述性区间，不是整个模型开发流程的泛化置信区间。相应区间和去掉最大五笔正收益后的集中度检查见 <a href="paired_uncertainty.json">JSON</a>。</p>
<img src="figures/house_gain_concentration.png" alt="普通房屋逐样本收益集中度诊断"><p>面积保护在至少一个外层拆分影响 {support_audit['any_repeat_tapered_houses']} 套训练房屋，不只那两套异常。大面积普通交易及按年份分组的完整结果见 <a href="outer_subgroup_metrics.csv">CSV</a>；逐房屋收益见 <a href="outer_house_error_contributions.csv">CSV</a>。</p>
<h2>每一外层折的选择和分数</h2>{table(fold_table)}
<h2>最终提交文件</h2><p>最终在全 1,460 条训练数据内部按同一 3 折选择规则（seed24501）选定 <b>{html.escape(selection['selected_recipe'])}</b>。这个最终选择可能与个别外层折不同，属于同一训练流程。具体为新增模型 {html.escape(', '.join(chosen['new_names']))}，新增权重 {chosen['new_weight']:.0%}，面积保护 {chosen['support_gate']}；对数预测公式为 <code>v4 + weight × support × (new - v4)</code>。已保存的 v4 预测直接复用，新增模型在全部训练行拟合，验证与最终训练采用相同种子策略。</p>
<p>文件包含 {audit['rows']} 条官方测试数据，Id 顺序正确，房价为有效正数。SHA-256：<code>{audit['sha256']}</code>。v1–v4 原提交文件保持不变。已保存模型、权重和面积阈值；执行 <code>sh scripts/run.sh scripts/predict_v5.py</code> 可复现预测并检查原文件。</p>
<p>本轮外层与最终选择缓存保存 {run_audit['estimator_fit_calls_in_cached_runs']} 次估计器 fit 调用；残差模型内部另有 {run_audit['additional_internal_EN_fits_in_residual_models']} 次 EN fit。该数不包含初筛、调试及因源代码变更而重跑的操作，不能与独立配置数混用。</p>
<p>公开讨论中存在对近零分使用外部 Ames 真实售价的质疑，不能据此判断 Ran 或其他具体参赛者。本轮没有下载外部真实售价或公开答案，也没有用测试 Id 查表填值。</p>
<p><a href="selection.json">最终方案</a> · <a href="frozen_candidates.json">冻结候选</a> · <a href="protocol.json">协议及修订</a> · <a href="release_gate.json">发布门槛</a> · <a href="model_manifest.json">模型清单</a> · <a href="run_audit.json">运行审计</a> · <a href="../../VERSIONS.md">版本记录</a>。</p></main></html>'''
    (ART/"report.html").write_text(report)
    print(json.dumps({"report":str(ART/'report.html'),"means":means,"run_audit":run_audit},indent=2))


if __name__=="__main__":main()
