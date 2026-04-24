import numpy as np 
from pipeline import final_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, fbeta_score
from config.core import config
from src.processing.loading import save_pipeline
from processing.loading import load_dataset

def run_training() -> None: 
    """Run the training process of the final model"""

    data = load_dataset(file_name=config.app_config.raw_data_file)

    ## we have to do the miscallenous cleaning

    X = data[config.model_settings.features]
    y = data[config.model_settings.target].map(config.model_settings.target_mappings)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        stratify=y, 
        test_size=config.model_settings.test_size,
        random_state=config.model_settings.random_state
    )

    final_pipeline.fit(X_train, y_train)

    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    recall_train = recall_score(y_train, y_train_pred)
    print(f"Recall (Train): {recall_train}")

    recall_val = recall_score(y_test, y_test_pred)
    print(f"Recall (Validation): {recall_val}")

    precision_train = precision_score(y_train, y_train_pred)
    print(f"Precision (Train): {precision_train}")

    precision_val = precision_score(y_test, y_test_pred)
    print(f"Precision (Validation): {precision_val}")

    fbeta_train = fbeta_score(y_train, y_train_pred, beta=2)
    print(f"F-Beta (Train): {fbeta_train}")

    fbeta_val = fbeta_score(y_test, y_test_pred, beta=2)
    print(f"F-Beta (Validation): {fbeta_val}")

    # save_pipeline() function 

    save_pipeline(pipeline_to_persist=final_pipeline)

if __name__ == "__main__": 
    run_training()

