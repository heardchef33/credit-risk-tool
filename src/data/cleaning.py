import pandas as pd 

def cleaning(df: pd.DataFrame): 

    print("Starting data cleaning")

    print("Grouping target variable to binary targets (Charged Off) and (Fully Paid) ...")

    df.drop(df.loc[(df['loan_status'] == 'Current') | (df['loan_status'] == 'In Grace Period') | (df['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid') | (df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off')].index, inplace=True)

    df.loc[df['loan_status'] == 'Late (16-30 days)', 'loan_status'] = 'Charged Off'

    df.loc[df['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 'Charged Off'

    df.loc[df['loan_status'] == 'Default', 'loan_status'] = 'Charged Off'

    print("Dropping loan status null values")

    df = df.loc[~df['loan_status'].isnull()]

    ## remove outliers (not doing it yet)

    ## dropping null values 

    columns_to_drop = []

    total_length = len(df)

    for column in df.columns: 

        if df[column].isnull().sum()/total_length >= 0.60: 

            df.drop(column, axis=1, inplace=True) # drop for the training set 

            columns_to_drop.append(column)
    
    print(f"A total of {len(columns_to_drop)} were dropped; with the columns being {columns_to_drop}")

    return df 

