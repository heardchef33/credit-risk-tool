import pandas as pd 
import time 

def loading(file_path): 

    try: 

        start_time = time.time()

        raw_data = pd.read_csv(file_path)

        elapsed_time = time.time() - start_time

        print(f"Data successfully read in {elapsed_time} seconds!")

        return raw_data

    except FileNotFoundError: 

        print("Error in reading the file. Please try again")

def sample(df): 

    return df.groupby('loan_status', group_keys=False).sample(frac=0.1, random_state=42)

if __name__ == '__main__': 

    loading("/Users/thananpornsethjinda/Desktop/credit-risk-modeling/data/accepted_2007_to_2018Q4.csv")

