"""Report TabPFN, structural-feature and training-context experiments.

Built with PriorLabs-TabPFN.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.run_experiments import load_data

ART=ROOT/"experiments/v6"


def collect_candidate_comparisons():
    """Combine completed seed5509 comparisons without selecting a new recipe."""
    frames=[]
    sources=sorted(ART.glob("comparison_*_seed5509.csv"))+sorted((ART/"variants").glob("*_comparison.csv"))
    for path in sources:
        frame=pd.read_csv(path)
        required={"tabpfn_weight","all","ordinary","normal"}
        assert required.issubset(frame.columns),path
        if path.parent.name=="variants":
            assert "variant" in frame and frame.variant.nunique()==1,path
            candidate=str(frame.variant.iloc[0])
        else:
            candidate=path.stem.removeprefix("comparison_").removesuffix("_seed5509")
        if "space" not in frame:frame["space"]="log"
        log_weights=frame.loc[frame.space.eq("log"),"tabpfn_weight"].to_numpy()
        assert all(np.isclose(log_weights,weight,rtol=0,atol=1e-12).any()
                   for weight in [0.,.25,.5,.75,1.]),path
        assert np.isfinite(frame[["tabpfn_weight","all","ordinary","normal"]]).all().all(),path
        frame["candidate"]=candidate
        frame["comparison_seed"]=5509
        frame["source_file"]=str(path.relative_to(ART))
        frames.append(frame[["candidate","tabpfn_weight","space","all","ordinary","normal","comparison_seed","source_file"]])
    columns=["candidate","tabpfn_weight","space","all","ordinary","normal","comparison_seed","source_file"]
    combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
    assert not combined.duplicated(["candidate","tabpfn_weight","space"]).any()
    return combined.sort_values(["candidate","space","tabpfn_weight"],ignore_index=True)


def main():
    selection=json.loads((ART/"selection.json").read_text());audit=json.loads((ART/"submission_audit.json").read_text())
    features=selection["features"];estimators=selection["n_estimators"];seed=selection["comparison_seed"]
    is_diverse=features=="diverse" and bool(selection.get("components"))
    if features=="diverse":assert is_diverse,"Diverse selection must record its components"
    oof=pd.read_csv(ART/f"oof_{features}_e{estimators}_seed{seed}.csv");train,_=load_data()
    assert np.array_equal(oof.Id,train.Id)
    comparison=pd.read_csv(ART/f"comparison_{features}_e{estimators}_seed{seed}.csv")
    candidate_comparison=collect_candidate_comparisons()
    candidate_comparison.to_csv(ART/"candidate_comparison.csv",index=False)
    y=oof.ylog.to_numpy();base=oof.v5_log.to_numpy();pfn=oof.tabpfn_log.to_numpy()
    base_weight=float(selection["weights_in_log_space"]["v5"])
    assert 0.<=base_weight<=1.
    blend=base_weight*base+(1.-base_weight)*pfn
    masks={"all":np.ones(len(train),bool),"ordinary_diagnostic":~((train.GrLivArea.to_numpy()>4000)&(train.SalePrice.to_numpy()<300000)),
        "Normal":train.SaleCondition.eq("Normal").to_numpy(),"area_gt_2500":train.GrLivArea.gt(2500).to_numpy()}
    for col in ["YrSold","SaleCondition"]:
        for value in sorted(train[col].unique()):
            mask=train[col].eq(value).to_numpy()
            if mask.sum()>=20:masks[f"{col}={value}"]=mask
    groups=[]
    for label,mask in masks.items():
        mse_gain=(base[mask]-y[mask])**2-(blend[mask]-y[mask])**2
        groups.append({"group":label,"rows":int(mask.sum()),"v5":float(np.sqrt(np.mean((base[mask]-y[mask])**2))),
            "TabPFN":float(np.sqrt(np.mean((pfn[mask]-y[mask])**2))),"v6":float(np.sqrt(np.mean((blend[mask]-y[mask])**2))),
            "fraction_houses_improved":float(np.mean(mse_gain>0)),
            "net_squared_error_gain":float(mse_gain.sum()),
            "net_squared_error_gain_without_top5":float(mse_gain.sum()-np.sort(mse_gain)[-5:].sum())})
    groups=pd.DataFrame(groups);groups.to_csv(ART/"subgroup_metrics.csv",index=False)
    ordinary_gain=groups.loc[groups.group.eq("ordinary_diagnostic"),"net_squared_error_gain_without_top5"].iloc[0]
    normal_gain=groups.loc[groups.group.eq("Normal"),"net_squared_error_gain_without_top5"].iloc[0]
    concentration_text=f'''<p>收益集中度检查：分别移除各组中平方误差改善最大的 5 套房屋后，剩余普通房屋的平方误差净改善为 <b>{ordinary_gain:+.6f}</b>，Normal 成交为 <b>{normal_gain:+.6f}</b>。正值表示剩余房屋合计改善，负值表示合计退步；这是诊断，不会从主要评分中删除任何房屋。表中同时列出每组误差下降的房屋比例。</p>'''
    details=train[["Id","SalePrice","GrLivArea","SaleCondition","Neighborhood","YrSold"]].copy()
    details["mse_gain_v6"]=(base-y)**2-(blend-y)**2
    details.sort_values("mse_gain_v6",ascending=False).to_csv(ART/"house_error_contributions.csv",index=False)
    folds=[]
    for fold in sorted(oof.fold.unique()):
        for group in ["all","ordinary_diagnostic","Normal"]:
            mask=(oof.fold.to_numpy()==fold)&masks[group]
            folds.append({"fold":int(fold),"group":group,"rows":int(mask.sum()),
                "v5":float(np.sqrt(np.mean((base[mask]-y[mask])**2))),"v6":float(np.sqrt(np.mean((blend[mask]-y[mask])**2)))})
    folds=pd.DataFrame(folds);folds.to_csv(ART/"fold_metrics.csv",index=False)
    figdir=ART/"figures";figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    subset=groups.head(3);x=np.arange(3)
    for name,color in [("v5","#b77b32"),("TabPFN","#8398b0"),("v6","#14826e")]:
        label="TabPFN representation bag" if name=="TabPFN" and is_diverse else name
        ax.plot(x,subset[name],"o-",label=label,color=color)
    ax.set_xticks(x,["All houses","Ordinary (diagnosis)","Normal transactions"])
    ax.set(ylabel="Pooled OOF log-RMSE",title="Matched historical folds: new TabPFN family and fixed blend")
    ax.spines[["top","right"]].set_visible(False);ax.legend();fig.savefig(figdir/"comparison.png",dpi=160);plt.close(fig)
    probe=json.loads((ART/"probe.json").read_text())
    online_path=ART/"online_result.json";online=json.loads(online_path.read_text()) if online_path.exists() else None
    status=f'用户反馈线上 {online["public_log_rmse"]:.5f}' if online else '尚未提交，Kaggle 分数未知'
    blend_summary_path=ART/"leaderboard_blend/summary.json"
    historical_blend_text=""
    if blend_summary_path.exists():
        historical_blend=json.loads(blend_summary_path.read_text())["released_submission"]
        historical_blend_text=f'''<h2>旧版本融合的收益上限</h2><p>已反馈线上分数与旧预测的几何关系给出，备用文件 <a href="../../{historical_blend['path']}">{historical_blend['path']}</a> 的估算 log-RMSE 为 {historical_blend['estimated_log_rmse']:.8f}，较 v5 仅约降低 <b>{historical_blend['estimated_improvement_vs_reported_v5']:.7f}</b>。因此本轮优先提交包含新模型的主 v6，旧预测融合文件作为备用。该估算依赖相同全量测试评分和输入分数的舍入前提，是利用线上反馈的计算；详见 <a href="leaderboard_blend/summary.json">融合计算与舍入范围</a>。</p>'''
    reproduction_path=ART/"reproduction_audit.json"
    reproduction_text=""
    if reproduction_path.exists():
        reproduction=json.loads(reproduction_path.read_text())
        verification_text=""
        if "verification_method" in reproduction:
            verification_text="先独立重载两套模型并对全部 1,459 行重建预测方向，再核对相同组件上下文，将已重建方向换权为最终三路等权并核验；这次换权无需新增 GPU 推理。"
        reproduction_text=f'''<p>已执行重载与预测核验。{verification_text}最大绝对对数预测差为 <b>{reproduction['max_abs_log_prediction_difference']:.8g}</b>，核验容差为 {reproduction['log_prediction_tolerance']:.8g}；旧提交文件校验通过。</p>'''
    if is_diverse:
        method_text=f'''当前主配置将两种表示等权集成：<b>physical 的 16 个内部成员</b>使用房屋结构特征，<b>quality 类别表示的 raw 的 4 个内部成员</b>将 OverallQual/OverallCond 作为类别。合计 {estimators} 个内部成员；两种表示各占 TabPFN 部分的一半，权重不按内部成员数量分配。设备为 <b>{selection['device']}</b>。'''
        if np.isclose(base_weight,1./3.,rtol=0,atol=1e-12):
            recipe_text='''主提交在对数空间将 <b>v5、physical/e16、quality 类别 raw/e4 三路各取 1/3</b>。表和图中的 TabPFN 指两种表示的等权平均，合计占主提交的 2/3，v5 占 1/3。'''
        else:
            recipe_text=f'''主提交在对数空间使用 <b>{base_weight:.2%} v5 + {(1.-base_weight)/2.:.2%} physical TabPFN + {(1.-base_weight)/2.:.2%} quality 类别 TabPFN</b>。表和图中的 TabPFN 指两种表示的等权平均。'''
        bag_diagnostic=json.loads((ART/"representation_bag_diagnostic.json").read_text())
        bag_results={float(row["quality_fraction_of_TabPFN"]):row for row in bag_diagnostic["results"]}
        physical_only=bag_results[0.];representation_bag=bag_results[.5]
        representation_text=f'''<h2>从初始半权融合到三路等权</h2><p>初案固定 TabPFN 总权重 50%。在这个权重下，两种表示平均后的全体 OOF log-RMSE 为 {representation_bag['all']:.8f}，physical 单表示为 {physical_only['all']:.8f}，差 {representation_bag['all']-physical_only['all']:.8f}；普通房屋从 {physical_only['ordinary']:.8f} 降为 {representation_bag['ordinary']:.8f}，Normal 从 {physical_only['normal']:.8f} 降为 {representation_bag['normal']:.8f}。因此保留两种互补表示，详见 <a href="representation_bag_diagnostic.json">早期表示比例诊断</a>。</p>
<p>后续比较粗权重与自然的三路等权，最终采用 v5、physical/e16、quality/e4 各 1/3，直接组合既有预测，无新增模型拟合。75% TabPFN 的总体分数更低，但相对半权融合的增量高度集中在两套极端房屋，因此未选作主提交。三路等权的普通房屋和 Normal 指标均优于半权初案。表示与权重选择参考了同一批历史 CV，存在适应选择偏差；线上效果仍需实际提交。</p>'''
        weight_diagnostic_path=ART/"weight_choice_diagnostic.json"
        if weight_diagnostic_path.exists():
            weight_diagnostic=json.loads(weight_diagnostic_path.read_text())
            chosen=next(row for row in weight_diagnostic["choices"] if row["name"]=="natural_three_equal")
            higher=next(row for row in weight_diagnostic["choices"] if row["name"]=="three_quarters")
            selected_group_gains=chosen["improvement_vs_v5"]["group_sensitivity"]
            representation_text+=f'''<p>相对半权初案，三路等权在普通房屋 {chosen['folds_improving_vs_half']['ordinary']}/5 折、Normal {chosen['folds_improving_vs_half']['normal']}/5 折改善；其调权净收益仍有 {chosen['increment_vs_half']['two_extreme_fraction_of_total_gain']:.1%} 来自两套极端房屋，75% 权重对应为 {higher['increment_vs_half']['two_extreme_fraction_of_total_gain']:.1%}。相对原 v5，分别去掉各组收益最大的 5 套后，普通房屋净平方误差收益仍为 {selected_group_gains['ordinary']['net_squared_error_gain_without_top5']:+.6f}，Normal 为 {selected_group_gains['normal']['net_squared_error_gain_without_top5']:+.6f}。完整粗权重、逐折和集中度诊断见 <a href="weight_choice_diagnostic.json">weight_choice_diagnostic.json</a>。</p>'''
    else:
        method_text=f'''当前主配置为 <b>{features}</b>，使用 {estimators} 个内部集成成员，设备 <b>{selection['device']}</b>。'''
        recipe_text=f'''本轮主交付采用对数空间的 <b>{1.-base_weight:.2%} TabPFN + {base_weight:.2%} v5</b>。'''
        representation_text=""
    def table(frame):return frame.to_html(index=False,float_format=lambda v:f"{v:.6f}",border=0)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TabPFN v6：新模型、结构特征与训练处理</title><style>body{{font:16px/1.8 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#24364b;background:#f3f6fa;margin:0}}main{{max-width:1100px;margin:25px auto;background:white;padding:30px;border-radius:12px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#286ba4}}</style><main>
<h1>v6：新模型、结构特征与训练处理并行探索</h1><p><b>Built with PriorLabs-TabPFN</b> · 官方公开 v2 回归权重及<a href="pretrained/LICENSE.txt">许可</a>已一并保存。</p>
<p>用户反馈 v5 实测为 <b>0.11595</b>。新文件：<a href="../../submission_v6.csv">submission_v6.csv</a>；{status}。</p>
<h2>本轮目标与研究线索</h2><p>目标是降低实际提交误差、提高排名。新模型、房屋结构特征和训练样本处理分别比较，候选及负结果全部保留。同学截图显示，旧四模型等权融合为 0.11963，后一次描述为“50% TabPFN + 50% 原集成”的提交为 0.11390。这提供了新模型方向；具体版本、代码、预处理和融合空间未知，两次记录不足以隔离各项变化的贡献。关于模型方法与其他排名路径的公开资料，见<a href="ranking_routes_research.md">排名路径研究</a>。</p>
<p>上一轮大量时间用于重复拟合传统模型；本轮所有 v5 同折参考都从已验证缓存重建，旧模型新增拟合次数为 <b>0</b>。准备独立 TabPFN 环境和锁文件，旧环境、模型及提交文件保持不变。</p>
<h2>主提交方法</h2><p>使用 TabPFN v2（Python 包 9.2.0）的官方公开回归权重，固定 revision 与 SHA-256。原始混合类型特征去掉 Id、SalePrice，MSSubClass 当类别处理；physical 配置额外构造面积、浴室、房龄、质量交互等房屋结构特征，预处理在各折训练部分拟合。目标为自然对数 SalePrice。{method_text}</p>
<p>{recipe_text}其他模型配置、25%/75%/100% 权重以及价格空间等权也有完整比较；本轮探索范围不限于这一配方。配置比较与扩展使用同一批历史房屋，因此存在方案选择偏差。</p>
{representation_text}
<h2>同折对比</h2><p>在全部 1,460 条训练数据上使用与保存的 v5 参考相同的 KFold 5 折 seed{seed}；所有验证房屋都保留。预处理及 TabPFN 训练上下文仅用各折训练行。<b>这些是已多轮使用的历史房屋，属于匹配缓存的回顾比较，不能当作新独立数据或 Kaggle 实测分数。</b>这里参考的是 v5 最终固定配方，不是 v5 先前报告的内层自动选择流程，两套数字不能混用。</p>
<img src="figures/comparison.png" alt="匹配历史拆分的v5、TabPFN与固定融合对比">{table(groups.head(4))}
{concentration_text}
<h2>主配置的融合诊断</h2>{table(comparison)}
<h2>完整候选与权重比较</h2><p>raw/e4 与 physical/e4 出现正收益后，本轮扩展到 physical/e16，并增加两个处理方向：仅在训练部分排除两套极端低价大房屋，以及将 OverallQual/OverallCond 改作类别表示。所有验证行保留，类别表示不施加质量单调约束。下面自动收集已完成的 seed5509 比较，包括每个候选的 0%/25%/50%/75%/100% TabPFN 粗权重、三路等权对应的 2/3 TabPFN 权重，以及已完成的价格空间诊断。完整 CSV：<a href="candidate_comparison.csv">candidate_comparison.csv</a>。</p>{table(candidate_comparison)}
{historical_blend_text}
<h2>每折与分组结果</h2>{table(folds)}{table(groups)}
<h2>真实运行时间和复现</h2><p>首次 n_estimators=1 单折探针：fit {probe['fit_seconds']:.2f} 秒，32 条预测 {probe['predict_32_seconds']:.2f} 秒，292 条预测 {probe['predict_292_seconds']:.2f} 秒。最终全量 fit 和 1,459 条测试预测共 {selection['elapsed_final_fit_predict_seconds']:.2f} 秒。环境首次安装时间单独记录，不混同为模型训练耗时。</p>
<p>提交文件共 {audit['rows']} 行，Id 顺序与官方测试集相同，房价为有效正数。SHA-256：<code>{audit['sha256']}</code>。保存原始公开预训练权重、其许可、实际训练上下文和精确配置，执行 <code>sh scripts/run_tabpfn.sh scripts/predict_v6.py</code> 可重建预测；GPU 浮点误差容差在重载审计中明确记录。</p>
{reproduction_text}
<p><a href="selection.json">方案</a> · <a href="protocol.json">初始方法与探索记录</a> · <a href="pretrained/manifest.json">预训练来源</a> · <a href="tabpfn_research.md">官方方法与版本研究</a> · <a href="ranking_routes_research.md">排名路径研究</a> · <a href="candidate_comparison.csv">完整候选比较</a> · <a href="reference/audit.json">旧预测缓存来源</a> · <a href="house_error_contributions.csv">逐房屋收益</a> · <a href="reproduction_audit.json">重载检查</a>。</p></main></html>'''
    (ART/"report.html").write_text(report)
    print(groups.head(4).to_string(index=False));print(ART/"report.html")


if __name__=="__main__":main()
