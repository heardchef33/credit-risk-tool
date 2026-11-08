"""Train the final model pipeline and save it to model/saved_models.

Run from the repository root:

    python -m src.train
"""
from loguru import logger
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config.core import config
from src.pipeline import final_pipeline
from src.processing.loading import load_dataset, save_pipeline


def run_training() -> None:
    """Run the training process of the final model"""

    data = load_dataset(file_name=config.app_config.raw_data_file)

    X = data[config.model_settings.features]
    y = data[config.model_settings.target].map(config.model_settings.target_mappings)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=config.model_settings.test_size,
        random_state=config.model_settings.random_state,
    )

    final_pipeline.fit(X_train, y_train)

    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    for split, y_true, y_pred in (("Train", y_train, y_train_pred), ("Validation", y_test, y_test_pred)):
        logger.info(f"Recall ({split}): {recall_score(y_true, y_pred):.4f}")
        logger.info(f"Precision ({split}): {precision_score(y_true, y_pred):.4f}")
        logger.info(f"F2 ({split}): {fbeta_score(y_true, y_pred, beta=2):.4f}")

    save_pipeline(pipeline_to_persist=final_pipeline)


if __name__ == "__main__":
    run_training()
