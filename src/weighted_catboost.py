"""Sample-weight-capable CatBoost adapter for the v2 robustness experiment."""
from src.house_prices import CatEstimator


class WeightedCatEstimator(CatEstimator):
    def fit(self, X, y, sample_weight=None):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            loss_function="RMSE", random_seed=self.seed, verbose=False,
            thread_count=4, allow_writing_files=False,
            cat_features=list(X.select_dtypes(exclude="number").columns),
            **(self.params or {}),
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self
