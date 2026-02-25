import pandas as pd 

import scipy.stats as stats

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

def mwu(charged_off: pd.DataFrame, fully_paid: pd.DataFrame):
    """
    Input: charged-off group and fully-paid group
    Returns: data splits containing subset of the features / columns to drop maybe 
    Purpose: reduce the feature space and remove statistically insignificant numerical variables
    """

    # Mann-Whitney U test to test for correlation between the continuous features and the categorical targets 

    # We establish the level of significance to be 5%

    # Let the charged-off group be denoted by C 
    # Let the fully paid group be denoted by F 

    # H0: F_median = C_median (median of the fully paid group is the same as the median of the charged off group)

    # H1: F_median != C_median (median of the fully paid group is the same as the median of the charged off group)

    # So, if p-value < 0.05, we reject the null hypothesis to include that the medians are different so then 
    # the correlation between that specific continuous feature and categorical target is significant 

    columns_to_drop_from_mwu = []

    for column in charged_off.select_dtypes(include=('float64')).columns: 
        stat, p_value = stats.mannwhitneyu(charged_off[column], fully_paid[column], alternative='two-sided')

        if p_value >= 0.05: 
            print(f"{column} to be dropped since p_value = {p_value}")
            columns_to_drop_from_mwu.append(column)

    return columns_to_drop_from_mwu


