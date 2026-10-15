# v5 方法检索笔记

检索日期：2026-10-15。目的：为当前线上最佳 0.11679 寻找可复现的基础模型和特征方向。下文的公开成绩只代表作者/页面报告，本项目尚未独立复现；历史百分位和名次不能当作当前排名。

## 优先验证的五个方向

1. **RBF SVR，补充现有 ElasticNet/CatBoost 的误差形态。** 作者[公开代码仓库](https://github.com/suzuran0y/house-price-regression-prediction)给出 `RobustScaler + SVR(C=20, epsilon=0.008, gamma=0.0003)`。可从这个小网格开始：C=10/20/40、gamma=0.0003/0.001/0.003、epsilon=0.008/0.02；只缩放连续特征，类别先用普通 one-hot，避免直接继承 v3 的稀有交叉。具体网格是本项目建议，不能视为公开验证结果。[官方 SVR 文档](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)支持 RBF 核，说明 C、epsilon、gamma 的作用；样本量仅 1460，精确核方法可行。

2. **Huber 梯度提升与平方误差配对比较。** 同一作者仓库包含 sklearn `GradientBoostingRegressor` 的 Huber 配方；可先试 depth=3/4、leaf=10/15、max_features=sqrt、learning_rate=0.03/0.05、2000–4000 棵树，保持同折及同一批预处理。此处是 sklearn 的 Huber GB，应与之前小 delta 的 CatBoost Huber 初始化失败区分。[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)说明 Huber 结合平方与绝对误差，训练初值由损失决定。稳健损失必须在普通房屋上也有净收益才可加入提交。

3. **二次核岭回归，检验平滑的连续交互。** 作者[公开方案](https://github.com/Pradnya1208/House-prices-prediction)提供多项式 KernelRidge；建议 degree=2、coef0=1/2.5、alpha=0.3/0.6/3 的小网格，同时保留缩放后的普通 one-hot。相对手写大量类别交叉，这能检验连续关系而不依赖稀有组合。核函数、正则参数见[官方 KernelRidge 文档](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)。这仍是待验证假设。

4. **训练折内目标编码替代稀有交叉。** [作者仓库](https://github.com/haywhy-ml/House-Prices-Advanced-Regression)报告对 Neighborhood 做 fold-safe target encoding，可作为线性/核模型的新输入。先仅编码 Neighborhood，再单独验证 Neighborhood×OverallQual，不在全训练集预先算均价。[官方 TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)的 `fit_transform` 使用内部交叉拟合；`fit(...).transform(...)` 会失去这一保护。外层验证也必须完全隔离；未知类别回退到训练目标均值。

5. **让融合权重通过额外的验证，而不只拟合一份 OOF。** [Serigne 原始 Kaggle notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)是历史 stacking 范例，但页面显示最佳 0.11924，不能以旧标题“Top 4%”证明它优于当前 v4。[另一作者仓库](https://github.com/goktani/house-prices-ensemble)自报多模型 OOF 0.10830 / 线上约 0.111，尚未由本项目复现。优先比较简单的 10%/25% 新模型加入比例，再考虑有强正则的非负权重；不同时搜索大量权重。若训练二级回归，需在外层验证内重新生成一级 OOF；仅对既有 OOF 再 KFold 并不能消除一级模型的跨折依赖。官方 [StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)区分一级全量拟合与元模型的交叉预测训练；[nested CV 示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)说明同数据选参再报分会产生乐观偏差。

## 必须同时看的证据

上述是研究建议，不是对本项目收益的结论。每个候选都应与 v4 在完全相同的验证行上比较：全体 RMSE、排除预先指定的 Id 524/1299 后 RMSE、逐房屋平方误差差、价格与面积分组、原本 292 行诊断集。所有验证房屋保留在总体指标中；“排除两条”只是附加诊断，不能把这个指标改名为比赛成绩。训练时若尝试异常样本权重/过滤，只能在当前训练折内实施，且单独做消融。新的随机折也不是全新的房屋，不能声称已经得到独立数据验证。

## 榜首接近零误差意味着什么

第一手 [Kaggle 讨论：How do some people have scores nearly 0.000?](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion/587297) 中，有参与者称检查近零分 notebook 时看到额外 Ames 数据，并有人推测复制/匹配测试真值。这些是参与者观察与推测，不是主办方逐项审计，也不能据此判断任意具体用户采用了什么方法。单凭 Ran 说“榜 1”，无法判定其实现方式。

因此，近零分榜首不能直接充当纯预测模型的可达到误差下限。本轮只借鉴训练策略、模型多样性和验证设计；未下载外部测试标签、公开答案提交或按 Id 查表覆盖代码，也不将通过排行榜反推某条测试房价当作模型改进。
