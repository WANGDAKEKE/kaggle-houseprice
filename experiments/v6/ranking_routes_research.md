# House Prices ranking routes: evidence and decisions

Checked 2026-10-15. Research only. No external test labels, solution CSVs, or original sale-price tables were downloaded or used.

## What the official sources establish

The competition overview credits Dean De Cock's Ames Housing dataset and specifies 79 explanatory variables, prediction of each test house's SalePrice, and RMSE between log predicted and log observed price. This is an indefinitely running educational competition with a rolling leaderboard and no cash prize, points, or medals. [Official overview](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/citation).

De Cock's original 2011 paper describes **2,930 residential sales from 2006–2010**, including sale prices, and explains that parcel identifiers could link properties to the municipal assessor's records. That establishes why external record matching is possible in principle. It does not itself verify a particular Kaggle test row or every preprocessing difference between the original and competition releases. The paper also distinguishes three extreme partial sales from two unusually large but more appropriately priced homes; this supports testing training-side treatment of those records, not removing difficult validation houses. [Original paper, Journal of Statistics Education](https://jse.amstat.org/v19n3/decock.pdf), pages 1–4.

The present Kaggle `train.csv` and `test.csv` total 2,919 rows, which is smaller than the original paper's count. Therefore an assumption that original record order, identifiers, or all target values transfer without an audit would be unjustified. We did not inspect external records to perform such an audit.

## What near-zero leaderboard evidence does and does not show

A competition discussion titled **“Why are the top 50 teams all scoring 0.00044？”** contains participant explanations involving a ground-truth solution and access to a complete original dataset. This is direct evidence that the matching/answer route is publicly discussed in this competition. It is participant testimony, not an official Kaggle judgment and not proof of what any specific present leaderboard entrant did. [Kaggle discussion 329836](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion/329836).

A score close to zero can result from submitting already known sale prices; such a score measures successful recovery/matching of evaluation records. It should be labelled separately from a model's out-of-sample predictive accuracy. Neither this evidence nor a rank alone warrants claiming that all strong scores used that route, or that Ran used it. The user's classmate's **0.11390** screenshot names a TabPFN blend and is a different kind of evidence from a nearly zero public result.

Kaggle's own documentation defines leakage to include ground truth entering the test evaluation and warns against chasing the public leaderboard. It also states that external data is allowed **only when the competition's rules specify it**. [Kaggle competition documentation](https://www.kaggle.com/docs/competitions).

## External-data permission remains unverified

The current competition-specific [Rules page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules) and [Data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) returned JavaScript shells with no readable rules content in the web reader. A browser fallback was also unavailable: the in-app browser was absent and Chrome automation could not load its request-header policy. Other competitions' external-data clauses are not evidence for this competition.

Accordingly, this investigation **does not establish that external sale-price matching is permitted**. Before obtaining or using external property records or target prices, read the live competition-specific rules through an available signed-in browser and check the external-data and submission provisions. If the text is ambiguous, host clarification is necessary before that route. Public accessibility or a published notebook is not by itself permission. This is a concrete missing rules fact; ordinary model development can continue while it is resolved. [Kaggle staff clarification on external data](https://www.kaggle.com/general/16691).

## Three practical paths

### 1. Add a different model family, then blend by held-out errors

Proceed now with the installed public **TabPFN v2** checkpoint, raw mixed-type features and log target. Start with the saved v5 folds and retain every validation record. Compare TabPFN alone, existing v5, and fixed 25% / 50% log-space blends. Record all-row, Normal-sale, and non-extreme-house errors, their paired differences, and test prediction correlations. A complementary predictor can improve the ensemble even when its standalone average is less competitive. This is the immediate highest-information experiment, without requiring a new model account.

If it helps, extend to a small nonnegative OOF combination of TabPFN, ElasticNet/SVR, CatBoost and a genuinely different strong LightGBM/XGBoost configuration, with meta-model weights fitted inside an outer split. This changes model diversity and ensemble construction rather than merely repeating tree depth searches. Do not promise a score from the classmate's screenshot: their checkpoint, features and blend space are unknown.

### 2. Test structure and training-population assumptions

Use paired ablations for (a) original raw features versus compact housing features, (b) training-only exclusion or robust handling of extreme partial sales, and (c) direct log-price regression versus a fold-fitted broad location/quality component plus residual model. Keep the existing all-row validation target and report ordinary-sale gains separately. The original paper provides a reason to test partial-sale handling; it does not prove which training treatment improves this Kaggle split. A failed earlier downweighting experiment does not establish that every distinct treatment will fail, but it does justify limiting the grid.

Choose a fixed, small experiment set, require improvement across the ordinary-sales comparison, and preserve a negative-result log. If available models remain highly correlated, expanding another tiny tree-parameter neighborhood is unlikely to add as much information as this structural comparison.

### 3. Investigate external information only after resolving rules

If the current rules explicitly permit it, explore genuinely additional public features or additional non-overlapping training observations first, with provenance and overlap checks. Direct recovery of an evaluation property's known historical sale price is a separate **record-matching route** and must be reported as such; its local modelling validation cannot be presented as the explanation of its leaderboard result. Do not claim this route is currently permitted, and do not download a “solution file” before resolving the rules. The research above explains the existence of the route without using it.

For the current round, paths 1 and 2 are actionable and fully preserve the existing official-data-only lineage. Path 3 has a specific evidence dependency, not a reason to pause current experiments.
