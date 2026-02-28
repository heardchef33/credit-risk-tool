import pandas as pd 
import numpy as np 

from src.data.loading import loading, sample
from src.data.cleaning import cleaning
from src.data.feature_engineering import dropping_constant_features, correlation_analysis, mwu, test_of_independence, mutual_information
from src.data.splitting import train_val_test_split
from src.data.preprocessing import initial_preprocessing, categorical_preprocessing


def data_preparation_pipeline(filepath: str, want_sample: bool, features_to_keep: int): 
    """
    input: filepath, boolean for sample, number of features to keep
    returns: data splits
    purpose: pipeline to orchestrate all data preparation step
    """
    print("Starting data preparation pipeline")
    print("""
    1. Loading Data 
    2. Cleaning (Removing Null Values and Irrelevant Columns)
    3. Splitting using Stratified Sampling
    3. Preprocessing + Feature Selection/Engineering
    """)

    columns_to_drop = []

    df = loading(file_path=filepath)

    if want_sample: 

        df = sample(df)
    
    cleaned_df = cleaning(df=df)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(df=cleaned_df)

    X_train, X_val, X_test = initial_preprocessing(X_train=X_train, X_val=X_val, X_test=X_test)

    X_train, X_val, X_test = dropping_constant_features(X_train=X_train, X_val=X_val, X_test=X_test)

    grouped = X_train.merge(y_train, how='inner', left_index=True, right_index=True)

    charged_off = grouped.loc[grouped['loan_status'] == 'Charged Off']

    fully_paid = grouped.loc[grouped['loan_status'] == 'Fully Paid']

    columns_to_drop.extend(correlation_analysis(X_train=X_train, X_val=X_val, X_test=X_test))
    columns_to_drop.extend(mwu(charged_off=charged_off, fully_paid=fully_paid))
    columns_to_drop.extend(test_of_independence(X_train=X_train, whole_group=grouped))

    X_train.drop(columns_to_drop, axis=1, inplace=True) 
    X_val.drop(columns_to_drop, axis=1, inplace=True) 
    X_test.drop(columns_to_drop, axis=1, inplace=True)

    X_train, X_val, X_test, cat_columns = categorical_preprocessing(X_train=X_train, X_val=X_val, X_test=X_test)

    X_train, X_val, X_test = mutual_information(k=features_to_keep, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, cat_columns=cat_columns)

    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == '__main__': 

    filepath = "/Users/thananpornsethjinda/Desktop/credit-risk-modeling/data/accepted_2007_to_2018Q4.csv"
    want_sample = True 
    features_to_keep = 13
    data_preparation_pipeline(filepath=filepath, want_sample=want_sample, features_to_keep=features_to_keep)
