import pandas as pd 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def initial_preprocessing(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame): 
    """
    Input: Raw data splits 
    Returns: standardised data splits and imputation to remove null values
    Purpose: To remove null values to enable for statistical tests and feature selection techniques 
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # imputation to remove null values 

    print("Imputating missing numerical values ...")

    NUMERICAL_COLUMNS = X_train.select_dtypes(include=('float64')).columns.to_list()

    imputer = SimpleImputer(strategy='median')
    X_train[NUMERICAL_COLUMNS] = imputer.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_val[NUMERICAL_COLUMNS] = imputer.transform(X_val[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = imputer.transform(X_test[NUMERICAL_COLUMNS])

    print("Numerical imputation complete!")

    CATEGORICAL_COLUMNS = X_train.select_dtypes(include=('object')).columns.to_list()

    print("Imputating missing categorical values ...")

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[CATEGORICAL_COLUMNS] = cat_imputer.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_val[CATEGORICAL_COLUMNS] = cat_imputer.transform(X_val[CATEGORICAL_COLUMNS])
    X_test[CATEGORICAL_COLUMNS] = cat_imputer.transform(X_test[CATEGORICAL_COLUMNS])

    print("Categorical imputation complete!")

    # scaling 
    print("Scaling numerical values using robust scaler ...")

    scale = RobustScaler()

    X_train[NUMERICAL_COLUMNS] = scale.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_val[NUMERICAL_COLUMNS] = scale.transform(X_val[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scale.transform(X_test[NUMERICAL_COLUMNS]) 

    print("Scaling complete!")

    return X_train, X_val, X_test

def categorical_preprocessing(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame): 
    """
    Input: featured engineered data splits 
    Returns: data splits with encoded categorical columns 
    Purpose: encoding categories for training and for mutual information
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    ORDINAL_COLUMNS = ['sub_grade']

    ONE_HOT_COLUMNS = ['term', 'home_ownership']

    print("One-hot encoding select categorical columns ...")

    enc = OneHotEncoder(drop='first')

    enc.fit(X_train[ONE_HOT_COLUMNS])

    ONE_HOT_ENCODED_COLUMNS = list(enc.get_feature_names_out(ONE_HOT_COLUMNS))

    X_train[ONE_HOT_ENCODED_COLUMNS] = enc.transform(X_train[ONE_HOT_COLUMNS]).toarray()

    X_val[ONE_HOT_ENCODED_COLUMNS] = enc.transform(X_val[ONE_HOT_COLUMNS]).toarray()

    X_test[ONE_HOT_ENCODED_COLUMNS] = enc.transform(X_test[ONE_HOT_COLUMNS]).toarray()

    print("One-hot encoding successful!")

    print("Ordinal encoding select categorical columns ...")

    ordinal_enc = OrdinalEncoder()

    ordinal_enc.fit(X_train[ORDINAL_COLUMNS])

    ORDINAL_ENCODED_COLUMNS = list(ordinal_enc.get_feature_names_out(ORDINAL_COLUMNS))

    X_train[ORDINAL_ENCODED_COLUMNS] = ordinal_enc.transform(X_train[ORDINAL_COLUMNS])

    X_val[ORDINAL_ENCODED_COLUMNS] = ordinal_enc.transform(X_val[ORDINAL_COLUMNS])

    X_test[ORDINAL_ENCODED_COLUMNS] = ordinal_enc.transform(X_test[ORDINAL_COLUMNS])

    print("Ordinal encoding successful!")

    X_train_final = X_train.select_dtypes(exclude=['str']) # change to string

    X_val_final = X_val.select_dtypes(exclude=['str'])

    X_test_final = X_test.select_dtypes(exclude=['str'])


    return X_train_final, X_val_final, X_test_final, ORDINAL_ENCODED_COLUMNS + ONE_HOT_ENCODED_COLUMNS# have to return ordinal columns and encoded as well 







