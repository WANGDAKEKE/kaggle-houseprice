"""Fold-local continuous target encoding as an alternative tree representation."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from src.house_prices_v2 import DomainFeatures


def build_model(spec, seed=42):
    # TargetEncoder.fit_transform internally cross-fits. Calling fit().transform()
    # on the training rows instead would leak each row's target into its features.
    encoder = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encode", TargetEncoder(target_type="continuous", smooth=spec.get("smooth", 20.), cv=5,
                                 shuffle=True, random_state=seed)),
    ])
    columns = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), make_column_selector(dtype_include="number")),
        ("categorical", encoder, make_column_selector(dtype_exclude="number")),
    ], sparse_threshold=0.)
    return Pipeline([
        ("features", DomainFeatures(rich=False, log_numeric=False)),
        ("encoding", columns),
        ("model", HistGradientBoostingRegressor(early_stopping=False, random_state=seed,
              max_iter=850, learning_rate=.035, max_leaf_nodes=spec.get("leaves", 7),
              min_samples_leaf=20, l2_regularization=8., loss="squared_error")),
    ])
