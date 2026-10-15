# v6 实现独立审查

审查范围：以下文件，另读取旧缓存重建脚本及参考审计；未修改训练源、未拟合模型、未启动 GPU。

- `scripts/optimize_v6.py`：SHA-256 `fa6339205723ef35a107452e1bc768cce63b9865577df544b3b9d92cf9746822`
- `scripts/predict_v6.py`：SHA-256 `c3483fc182c3574facada7051780aa2db3c4b978fb6e1b85bac61d40c0f2fd05`
- `src/house_prices_v6.py`：SHA-256 `5c850b5376c4a93c6f5f1cf17c84c752f286e322cd2b0fb3e7f9a0aa96989682`
- `scripts/report_v6.py`：SHA-256 `5ab5cd75203b95792504cb81c249e26cff1f6b7defcf133f8db2791b0696483e`
- `requirements-tabpfn.lock.txt`：SHA-256 `bf700d1a9c142db27bda81acbda891b80c133d33c2c0f08a134bf7d1ea91f307`

## 当前结果的核验

- 当前完成的 raw / e4 / seed5509 缓存中，1,460 个 Id 与官方训练数据逐项一致；fold_ids 与参考 CSV 及重新生成的 KFold 5 折一致，预测均有限。
- 纯 CPU 特征构造检查：raw 为 79 列、44 个类别列；physical 为 112 列、44 个类别列。两者训练/测试列顺序一致，均排除了 Id、SalePrice；MSSubClass 为类别，原始字符串 None 保留，类别缺失转为 np.nan，数值列未生成无穷。检查仅构造特征，未调用 LocalTabPFN.fit/predict。
- `.venv-tabpfn/bin/python -m pip check` 返回 No broken requirements found。独立锁文件与旧环境隔离；没有修改旧 requirements.lock.txt。
- 参考重建逐折核对了训练/预测 Id、旧缓存签名及折号，匹配全量训练集 seed5501 / 5509 的 KFold。所用 v5 是最终固定配方的回顾 OOF，而不是先前内层自动选择流程；报告已说明此区别。
- 未发现需要重跑已经完成 raw CV 的实际标签泄漏、行错位或缺失值转换 bug。

## 改配方前必须同步处理

1. `optimize_v6.finalize`、`predict_v6.main` 和 `report_v6.main` 均硬编码 50%/50% 对数融合；selection 的模型名、权重、指标及文案同样固定。如改配比、混合多个 TabPFN 特征版本或改融合空间，必须让三个入口使用同一份配方配置，否则提交、重载核验和报告将描述不同预测。当前固定 50% 配方本身没有因此出错。
2. 当前 CV 指纹哈希包含 house_prices_v6.py、train.csv、独立锁文件及配置，但未包含继承的 src/house_prices.py；physical 实际依赖其 Features。若未来变更父特征而不改变 v6 包装器，旧 physical 缓存会被误用。当前父源冻结，故这不是现有 CV 的结果错误。添加严格指纹时可记录迁移审计，而不要为纯元数据修复重复训练。

## 其他记录

- CV 恢复执行时 `seconds` 只统计本次新跑折的耗时，完整缓存命中时不新增时间；若用于总算力报告，应另留跨会话累计耗时。此项不影响预测。
- 缓存加载分支目前仅判断签名，没有再次核对缓存自身 ids/fold_ids。此次已独立核验完成 raw 缓存正确；后续入口可补断言，避免未来外部文件改动引发静默行错位。
- GPU 重载仍需要实际运行 predict_v6.py 形成 reproduction_audit；静态审查不能替代该检查，也不承诺 5e-5 容差必然通过。
- 当前50%协议是在用户后续允许为排名调整配方之前写的；后续更改可以如实标为基于历史 CV 的方案选择，不必把旧协议当作禁止优化的限制。
