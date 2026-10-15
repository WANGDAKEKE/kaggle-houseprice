"""Audit v3's regression and reproduce a fixed, leaderboard-informed v2/v3 blend.

No new training or test labels. Known leaderboard scores are used explicitly;
the resulting score estimate is NOT an independent validation measurement.
"""
from pathlib import Path
import sys, os, json, hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.optimize_v2 import partitions
from scripts.optimize_v3 import SOURCE_HASH
from scripts.run_experiments import load_data, validate_submission

ART = ROOT / "experiments/v4"
SEEDS = [42, 2026, 137, 2701, 901]
WEIGHT_V3 = .5  # Fixed simple blend; do not tune a fine grid to displayed scores.
HASHES = {
    "submission.csv": "210223272bc07d8c7d483aadc47a9831faa248ca9c5c633699d12ebea9b3e3db",
    "submission_v2.csv": "66c2d66ab1b9fe43cfd0bc68229a9881c1181fb87115e5fad6350d363972eb6a",
    "submission_v3.csv": "f9787b96b898cb3ace1eea03bb79bffc56fb2967d34649155508b61d3f02b4de",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (ART / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rmse(target, prediction):
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def blend_rmse(score_a, score_b, distance_squared, weight):
    return np.sqrt((1-weight)*score_a**2 + weight*score_b**2
                   - weight*(1-weight)*distance_squared)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    for name, expected in HASHES.items():
        assert digest(ROOT / name) == expected, f"Frozen input changed: {name}"
    train, test = load_data()
    dev, _ = partitions(train)
    ids = train.Id.iloc[dev].to_numpy()
    target = np.log(train.SalePrice.to_numpy())[dev]
    config = json.loads((ROOT / "experiments/v3/specifications.json").read_text())
    names = ["ref_v2_linear", "ref_v2_cat", "en_inter_size_scaled", "cat_d5_onehot"]
    predictions = {}
    for name in names:
        columns = []
        for seed in SEEDS:
            path = ROOT / "experiments/v3/cv" / f"dev_{name}_seed{seed}.npz"
            saved = np.load(path)
            signature = hashlib.sha256((SOURCE_HASH + json.dumps({
                "name": name, "spec": config[name], "seed": seed,
                "rows": dev.tolist(), "prefix": "dev"}, sort_keys=True)).encode()).hexdigest()
            assert str(saved["signature"]) == signature, path
            assert np.array_equal(saved["ids"], ids)
            assert np.isfinite(saved["prediction"]).all()
            reference_folds = np.load(ROOT / "experiments/v3/cv" / f"dev_ref_v2_linear_seed{seed}.npz")["fold_ids"]
            assert np.array_equal(saved["fold_ids"], reference_folds)
            columns.append(saved["prediction"])
        predictions[name] = np.column_stack(columns)
    old_linear, old_cat, new_linear, new_cat = [predictions[n] for n in names]
    candidates = {
        "v2": (old_linear + old_cat) / 2,
        "v3": (new_linear + new_cat) / 2,
        "new_linear_old_cat": (new_linear + old_cat) / 2,
        "old_linear_new_cat": (old_linear + new_cat) / 2,
    }
    candidates["v4_half_blend"] = (candidates["v2"] + candidates["v3"]) / 2
    masks = {"all_1168": np.ones(len(dev), bool),
             "without_two_extreme_houses_diagnostic_only": ~np.isin(ids, [524, 1299])}
    for col in ["SaleCondition", "YrSold", "OverallQual"]:
        for val in sorted(train[col].unique()):
            mask = train.iloc[dev][col].eq(val).to_numpy()
            if mask.sum() >= 20:
                masks[f"{col}={val}"] = mask
    comparisons = []
    for group, mask in masks.items():
        for name, p in candidates.items():
            scores = [rmse(target[mask], p[mask, i]) for i in range(len(SEEDS))]
            comparisons.append({"group": group, "rows": int(mask.sum()), "model": name,
                "mean_split_log_rmse": float(np.mean(scores)),
                **{f"seed_{s}": scores[i] for i, s in enumerate(SEEDS)}})
    results = pd.DataFrame(comparisons)
    results.to_csv(ART / "paired_comparisons.csv", index=False)
    a, b = candidates["v2"], candidates["v3"]
    gain = np.mean((a-target[:, None])**2 - (b-target[:, None])**2, axis=1)
    details = train.iloc[dev][["Id", "SalePrice", "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition"]].copy()
    details["mean_squared_error_gain_v3"] = gain
    details.sort_values("mean_squared_error_gain_v3", ascending=False).to_csv(ART / "house_error_contributions.csv", index=False)
    # Paired house-level resampling; the five repeats are NOT independent houses.
    rng = np.random.default_rng(61527)
    bootstrap = {}
    for label in list(masks)[:2]:
        values = gain[masks[label]]
        means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(5000)])
        bootstrap[label] = {"mean_mse_gain_v3": float(values.mean()),
            "paired_house_bootstrap_95pct_interval": np.quantile(means, [.025, .975]).tolist()}
    diagnosis = {"validation_role": "retrospective diagnosis on reused development houses; not fresh validation",
        "split_seeds": SEEDS, "original_rows": len(dev), "validation_rows_removed_for_primary_metric": 0,
        "diagnostic_extreme_ids": [524, 1299], "v3_total_squared_error_gain": float(gain.sum()),
        "two_extreme_houses_gain": float(gain[np.isin(ids, [524, 1299])].sum()),
        "two_house_share_of_net_gain": float(gain[np.isin(ids, [524, 1299])].sum()/gain.sum()),
        "fraction_houses_with_positive_gain": float(np.mean(gain > 0)),
        "bootstrap": bootstrap,
        "conclusion": "V3's net development gain is more than explained by two extreme houses; the remainder slightly regresses. This is a diagnostic finding, not proof of why unseen test errors changed."}
    write_json("regression_diagnosis.json", diagnosis)

    v2, v3 = pd.read_csv(ROOT / "submission_v2.csv"), pd.read_csv(ROOT / "submission_v3.csv")
    for frame in [v2, v3]:
        validate_submission(frame, test)
    assert np.array_equal(v2.Id, v3.Id)
    x, z = np.log(v2.SalePrice.to_numpy()), np.log(v3.SalePrice.to_numpy())
    d2 = float(np.mean((x-z)**2))
    online = json.loads((ROOT / "artifacts/kaggle_results.json").read_text())["submissions"]
    records = {r["version"]: r for r in online}
    for version, name in [("v2", "submission_v2.csv"), ("v3", "submission_v3.csv")]:
        assert records[version]["submission_sha256"] == HASHES[name]
    s2, s3 = [records[v]["public_log_rmse"] for v in ["v2", "v3"]]
    estimate = float(blend_rmse(s2, s3, d2, WEIGHT_V3))
    # Five-decimal displayed score rounding, not a statistical confidence interval.
    radius = .000005
    bounds = [float(blend_rmse(s2+sign*radius, s3+sign*radius, d2, WEIGHT_V3)) for sign in [-1, 1]]
    # Check the RMSE identity against actual known development targets at all weights.
    max_formula_error = 0.
    for i in range(len(SEEDS)):
        for weight in [0., .25, .5, .75, 1.]:
            inferred = blend_rmse(rmse(target, a[:, i]), rmse(target, b[:, i]),
                np.mean((a[:, i]-b[:, i])**2), weight)
            measured = rmse(target, (1-weight)*a[:, i]+weight*b[:, i])
            max_formula_error = max(max_formula_error, abs(float(inferred)-measured))
    assert max_formula_error < 1e-12
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp((1-WEIGHT_V3)*x+WEIGHT_V3*z)})
    validate_submission(submission, test)
    destination = ROOT / "submission_v4.csv"
    submission.to_csv(destination, index=False)
    reread = pd.read_csv(destination)
    validate_submission(reread, test)
    assert np.allclose(np.log(reread.SalePrice), (x+z)/2, atol=1e-12, rtol=0)
    estimate_info = {"status": "algebraic estimate; not an observed Kaggle submission score",
        "expected_log_rmse": estimate, "displayed_input_rounding_bound": bounds,
        "input_score_precision": 5, "score_sources": "user reports for exact frozen v2/v3 files",
        "v2_score": s2, "v3_score": s3, "test_log_prediction_distance": float(np.sqrt(d2)),
        "formula": "E(w)^2=(1-w)*E2^2+w*E3^2-w*(1-w)*mean((log(p2)-log(p3))^2)",
        "required_assumptions": ["reported scores belong to the hashed files",
            "both scores evaluate the same complete 1459-row test set with equal row weights",
            "metric is RMSE of log(SalePrice), as described in the supplied official overview",
            "displayed scores rounded to nearest 0.00001"],
        "coverage_source": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard",
        "coverage_observation_date": "2026-10-15",
        "coverage_visible_text": "This leaderboard is calculated with all of the test data.",
        "interpretation": "Leaderboard-informed combination of existing predictions; not evidence of independent model generalization or access to any individual test label.",
        "identity_numerical_check_max_abs_error": max_formula_error}
    write_json("score_estimate.json", estimate_info)
    selected = results.query("group == 'all_1168' and model == 'v4_half_blend'").iloc[0]
    selection = {"version": "v4", "method": "fixed geometric mean of v2 and v3 prices",
        "weights_in_log_space": {"submission_v2.csv": .5, "submission_v3.csv": .5},
        "selection_basis": "explicit leaderboard feedback and complementary prediction errors; no additional model training",
        "public_log_rmse": None, "estimated_public_log_rmse": estimate,
        "reused_five_split_development_mean": float(selected.mean_split_log_rmse),
        "new_independent_validation": False, "new_training_fits": 0,
        "test_labels_used": False, "input_sha256": HASHES,
        "future_training_gate": "Compare matched component ablations with whole-sample, ordinary-sale, extreme-house and sale-year diagnostics; keep all validation rows. Use a small predeclared candidate set in outer CV, with tuning only in inner folds; reuse of this dataset must still be disclosed."}
    write_json("selection.json", selection)
    audit = {"rows": len(submission), "columns": list(submission.columns),
        "positive_finite": True, "id_order_matches_official_test": True,
        "sha256": digest(destination), "preserved_input_sha256": HASHES,
        "min_price": float(submission.SalePrice.min()), "max_price": float(submission.SalePrice.max()),
        "submitted_to_kaggle": False, "generation_command": "sh scripts/run.sh scripts/optimize_v4.py"}
    write_json("submission_audit.json", audit)
    leaderboard = {"observed_date": "2026-10-15", "url": estimate_info["coverage_source"],
        "access": "public page rendered in browser, logged out", "coverage_text": estimate_info["coverage_visible_text"],
        "visible_table_total_teams": 4209, "visible_zero_score_entries": 18,
        "visible_score_examples": {"rank_44": .09531, "rank_45": .10667, "rank_49": .10865},
        "limit": "Only the first 49 entries were read. Scores alone do not reveal the methods used. Our exact rank was not determined."}
    write_json("leaderboard_observation.json", leaderboard)
    plot_report(results, details, diagnosis, estimate_info, audit)
    snapshot = ROOT / "versions/v4"
    snapshot.mkdir(exist_ok=True)
    for name in ["selection.json", "score_estimate.json", "submission_audit.json", "regression_diagnosis.json"]:
        (snapshot / name).write_bytes((ART / name).read_bytes())
    (snapshot / destination.name).write_bytes(destination.read_bytes())
    for name, expected in HASHES.items():
        assert digest(ROOT / name) == expected
    print(json.dumps({"submission": str(destination), "audit": audit, "estimate": estimate_info,
        "all_rows_comparison": results.query("group == 'all_1168'").to_dict("records")}, indent=2))


def plot_report(results, details, diagnosis, estimate, audit):
    online_path = ART / "online_result.json"
    online = json.loads(online_path.read_text()) if online_path.exists() else None
    online_status = (
        f'用户已提交 v4，反馈实测 <b>{online["public_log_rmse"]:.5f}</b>，目前为已提交最佳版本。'
        if online else 'v2 仍是目前实际提交的最佳版本。v4 尚未提交。'
    )
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, group, title in zip(axes, ["all_1168", "without_two_extreme_houses_diagnostic_only"],
            ["All development houses", "Excluding two extremes (diagnosis only)"]):
        subset = results.loc[(results.group == group) & results.model.isin(["v2", "v3", "v4_half_blend"])]
        ax.bar(subset.model, subset.mean_split_log_rmse, color=["#d79547", "#3a91aa", "#16846d"])
        ax.set(title=title, ylabel="Mean of five OOF log-RMSE values", ylim=(.105, .125))
        for i, v in enumerate(subset.mean_split_log_rmse):
            ax.text(i, v+.0003, f"{v:.5f}", ha="center", fontsize=9)
    fig.savefig(ART / "figures/paired_diagnosis.png")
    plt.close(fig)
    table = results.loc[results.group.isin(["all_1168", "without_two_extreme_houses_diagnostic_only", "SaleCondition=Normal"]),
        ["group", "rows", "model", "mean_split_log_rmse"]].to_html(index=False, float_format=lambda v: f"{v:.6f}")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>House Prices v4：回退诊断与固定融合</title><style>body{{font:16px/1.8 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#233148;background:#f4f7fb;margin:0}}main{{max-width:1050px;margin:30px auto;background:white;padding:30px;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}a{{color:#226da3}}</style><main>
<h1>v4：先解释 v3 的退步，再减少单一方案的偏差</h1>
<p>用户反馈：v2 = <b>0.11741</b>，v3 = <b>0.11770</b>。{online_status}</p>
<h2>为什么之前的对比容易误导</h2><p>原来的五次拆分都使用同一批 1,168 套房屋。v3 对 Id 524、1299 两套极大面积低价房屋的平方误差改善，占净改善的 <b>{diagnosis['two_house_share_of_net_gain']*100:.1f}%</b>；其余房屋合计略有退步。普通成交房屋也略有退步。这能够解释本地指标为什么容易被极少数样本主导，但不能证明测试集退步的具体原因。</p>
<p>下表加入“新线性模型 + 旧树”和“旧线性模型 + 新树”的匹配消融，帮助区分组件影响。剔除两套房屋只用于诊断；主要指标、训练文件、提交文件均没有删除或针对 Id 覆盖任何样本。</p>
<img src="figures/paired_diagnosis.png" alt="全部样本与去掉两个极端样本的诊断对比">{table}
<p>逐房屋贡献：<a href="house_error_contributions.csv">CSV</a>；所有拆分及按年份、质量、交易状态的比较：<a href="paired_comparisons.csv">CSV</a>；5,000 次按房屋配对 bootstrap 的 MSE 差区间：<a href="regression_diagnosis.json">JSON</a>。这些均是复用开发集的回顾分析，不能当作新的独立验证。</p>
<h2>v4 的提交方法</h2><p><code>SalePrice_v4 = exp(0.5 * log(SalePrice_v2) + 0.5 * log(SalePrice_v3))</code>。固定等权融合，保留两版互补部分，不为极小的估算差异细调权重。本轮没有新增训练拟合。</p>
<p>官方<a href="{estimate['coverage_source']}">排行榜</a>的可见说明明确写着使用全部测试数据。对相同测试集的 log-RMSE，已知两版分数和两版对数预测的均方距离，便可按以下恒等式估算融合分数：</p>
<p><code>E(w)^2 = (1-w)E2^2 + wE3^2 - w(1-w)mean((log(p2)-log(p3))^2)</code></p>
<p>取 w=0.5，估算为 <b>{estimate['expected_log_rmse']:.8f}</b>。若两份反馈分数按五位小数四舍五入，输入舍入对应范围约为 <b>{estimate['displayed_input_rounding_bound'][0]:.8f}–{estimate['displayed_input_rounding_bound'][1]:.8f}</b>。这个范围只是舍入误差界限，不是置信区间，也不是实际提交分数。公式已用有真实标签的开发集逐拆分数值核验。</p>
<p>估算要求两份分数确实对应保存的 v2/v3 文件，且都按相同的完整测试集等权计算 log-RMSE。v4 明确使用线上反馈，不能声称它是完全不看榜单得到的模型改进；没有读取任何单套测试房屋的标签。官网显示若干零分，仅凭分数无法判断其方法，本轮没有使用这些提交或外部答案。</p>
<h2>文件与下一轮比较</h2><p><a href="../../submission_v4.csv">submission_v4.csv</a>：{audit['rows']} 行，Id 与官方测试集顺序一致，价格正数且有限。SHA-256：<code>{audit['sha256']}</code>。v1、v2、v3 文件校验值均保持不变。</p>
<p>后续训练先固定少量候选，在外层验证中比较匹配消融，参数只在内层选择；同时报告全样本、普通交易、异常房屋和年份分组结果。需要明确承认这些数据已有多轮使用记录。嵌套验证避免同一分数同时选参数和评价方案的原则，见 <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">scikit-learn 官方说明</a>。</p>
<p>复现：<code>sh scripts/run.sh scripts/optimize_v4.py</code>。<a href="selection.json">方案</a> · <a href="score_estimate.json">估算与前提</a> · <a href="submission_audit.json">文件校验</a> · <a href="leaderboard_observation.json">榜单观察</a> · <a href="../../VERSIONS.md">版本记录</a>。</p></main></html>'''
    (ART / "report.html").write_text(report)


if __name__ == "__main__":
    main()
