import numpy as np
from sklearn._config import set_config
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.config.core import config

set_config(transform_output="pandas")

impute = ColumnTransformer(
    [
        ("numerical_imputation", SimpleImputer(strategy='median'),
         config.model_settings.numerical_features),
        ("categorical_imputation", SimpleImputer(strategy='most_frequent'),
         config.model_settings.categorical_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

encode = ColumnTransformer(
    [
        ("one_hot_encoding",
         OneHotEncoder(drop='first', sparse_output=False, dtype=np.int64),
         config.model_settings.categorical_features_to_encode_one_hot),
        ("ordinal_encoding",
         OrdinalEncoder(dtype=np.int64),
         config.model_settings.categorical_features_to_encode_ordinally),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

scale = make_column_transformer(
    (RobustScaler(), config.model_settings.numerical_features),
    remainder='passthrough',
    verbose_feature_names_out=False
)

model = XGBClassifier(
    n_estimators=config.model_settings.n_estimators,
    random_state=config.model_settings.random_state,
    scale_pos_weight=config.model_settings.scale_pos_weight,
    booster=config.model_settings.booster,
    learning_rate=config.model_settings.learning_rate,
    max_depth=config.model_settings.max_depth,
    min_child_weight=config.model_settings.min_child_weight,
    objective=config.model_settings.objective
    )

final_pipeline = Pipeline([
            ("imputation",impute),
            ("encoding",encode),
            ("feature_scaling", scale),
            ("xgboost_model", model)
        ])
