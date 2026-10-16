本版记录最终采用的 v6 成绩 **0.11212**（2026-10-16，用户反馈），较 v5 的 0.11595 降低 3.30%。

模型与提交文件和 `v6.0.0` 完全一致；本次只补充线上成绩、最终报告、分享材料和平台安装参考。

最终配方在对数空间将 v5、住宅结构特征 TabPFN/e16、质量类别特征 TabPFN/e4 各取 1/3。所有最终模型保留完整训练数据；本地历史五折分数与线上实测分别记录。

附件：

- `kaggle-houseprice-v6.1.0.zip`：完整代码、官方原始数据、模型上下文、公开预训练权重及许可、历史实验、最终提交与报告。解压后先读 `docs/SHARING_GUIDE.md`。
- `house_prices_v6_final_report.pdf`：3 页最终报告，可直接转发。
- `house_prices_v6_final_report.html`：自包含 HTML，图片已内嵌。
- `share_manifest.json`：版本、ZIP 完整性与文件校验信息。

私有仓库的网页及附件需要仓库访问权限；所有者可直接转发下载好的 ZIP/PDF。Python 环境和登录凭据不在代码包中。

Built with PriorLabs-TabPFN。模型许可随代码保留。
