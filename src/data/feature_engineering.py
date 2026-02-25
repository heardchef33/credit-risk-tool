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

def correlation_analysis(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Input: data splits 
    Returns: data splits containing subset of the features from correlation analysis (found in the /notebooks)
    Purpose: reduce the feature space and multicollinear features 
    """

    print("Starting correlation analysis ...")

    corr_columns_drop = [
        'loan_amnt', 
        'funded_amnt_inv',
        'open_acc',
        'pub_rec_bankruptcies',
        'tot_hi_cred_lim',
        'avg_cur_bal',
        'total_bc_limit',
        'num_actv_bc_tl',
        'num_rev_tl_bal_gt_0',
        'num_bc_tl',
        'num_op_rev_tl',
        'percent_bc_gt_75',
        'total_il_high_credit_limit'
        ]
    
    return corr_columns_drop


