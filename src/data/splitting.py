import pandas as pd 

from sklearn.model_selection import train_test_split

def train_val_test_split(df: pd.Dataframe): 

    X = df.drop('loan_status', axis=1)

    y = df['loan_status']

    # stratification for now; will stratify again later
    # splitting into train, validation and test sets 

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    X_validation, X_test, y_validation, y_test = train_test_split(X_valid_test, y_valid_test, stratify=y_valid_test, test_size=0.50, random_state=42)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

