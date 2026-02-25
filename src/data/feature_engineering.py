import pandas as pd 

from feature_engine.selection import DropConstantFeatures



def dropping_constant_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame): 
    """
    Input: data splits after scaling and imputation 
    Returns: data splits containing subset of the features 
    Purpose: reduce the feature space and drop constant and quasi-constant features
    """

    print("Dropping constant and quasi-constant features ...")

    sel = DropConstantFeatures(tol=0.95)

    sel.fit(X_train)

    X_train = sel.transform(X_train)
    X_val = sel.transform(X_val)
    X_test = sel.transform(X_test)

    print(f"{len(sel.features_to_drop_)} Constant and Quasi-Constant features dropped; they are: {sel.features_to_drop_}; ")

    return X_train, X_val, X_test


