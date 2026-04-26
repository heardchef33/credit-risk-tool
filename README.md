# Credit-Risk-Tool
Full End-to-End Machine Learning project to predict loan defaults
## Overview

This project provides a complete credit risk prediction system consisting of:
- **Backend API**: FastAPI-based REST API for making predictions hosted by Render 
- **Frontend**: Streamlit web interface for interactive predictions hosted by Streamlit Cloud 

Also included is the whole model development process from: 
- Analysing the Data and carrying out EDA 
- Feature Selection and Feature Engineering 
- Model Selection and Hyperparameter Finetuning 

## Inspiration and Purpose of Project 

My goal for this project was to simulate the role of a data scientist/machine learning engineer when given a machine learning problem and to improve my data science and software engineering skills. 

Core skills learnt from this project: 
- Dealing with class imbalance through choosing the right metrics and cost-sensitive learning by considering the inverse of the class distributions
- Avoiding data leakage during cross validation by implementing preprocessing and feature selection within each iteration 
- Using different industry-adopted tools such as `MLFlow` for experiment tracking and `Optuna` for hyperparameter fine-tuning 

## Dataset and Problem Information 

This dataset contains lending loan data from Lending Club, a peer-to-peer (P2P) lending platform which was obtained from Kaggle. (https://www.kaggle.com/datasets/wordsforthewise/lending-club)

The aim was to create a model that can predict whether or not a loan will likely default or be fully paid based on various factors. In this scenario, the loans that will default or be charged off is marked as the positive class (1) while those that will be fully paid were marked as (0) since we are interesting in identifying whether or not a loan will default.

Since accepting a loan that will default is far worse than rejecting a loan that will be fully paid, reducing false positives is more critical

## Modeling and Development Process 
The development of the model can be found in the `/notebooks ` folder. 
Each notebook is numbered from 01 to 07 with the contents of each notebook coming in sequential order. 
A brief summary of each notebook will be written below. For more information, consider looking to each notebook. 

- `01_EDA`
This notebook involved understand the shape, spread and the distribution of the data and finding out whether or not the data fits underlying assumptions (eg. whether or not the data is normal). On top of that, we also looked the target class distributions in this binary classification problem. 

- `02_domain_analysis`
The dataset contains over 130 features with each feature being a indicator of whether or not the loan has defaulted or was fully paid. However, some features were irrelevant to the domain and posed concerns with regards to data leakage. Therefore analysis had to be done with respect to existing credit risk frameworks to remove redundant and noisy features. 

- `03_miscalleneous_cleaning`
This notebook involved basic cleaning of the dataset and preparing the dataset for modeling through dropping features we identified as redundant in the previous step, dropping null values. An important step of the development process covered in this notebook was reducing the target categories to only 2 main classes as the targets in the intial dataset had noisy extras. 

- `04_feature_engineering`
This notebook involved carrying out preprocessing on the data and carrying out feature selection to reduce the feature space to only 10 features using the mutual information criterion. 

- `05_model_selection`
There were a variety of models to choose from for this problem. So, this notebook involves the selection of the best performing model against a baseline (which we also establish). Apart from that, we define the evaluation metrics used for this machine learning problem. 

- `06_model_finetuning`
Once we have chosen the right model, we carried out bayesian optimisation to find the parameters that gave us the best model.

- `07_final_model_pipeline`
This notebook involved training the final best model on the whole dataset.

## Model Information 

The final model chosen for this problem was `XGBoost` as it showed optimal balance between generalisability, maximising recall and computational time. 

The model uses the following 10 features for predictions:
- `int_rate`: Interest rate (%)
- `fico_range_high`: FICO score range high
- `inq_last_6mths`: Inquiries in last 6 months
- `open_il_12m`: Open installment accounts (12 months)
- `acc_open_past_24mths`: Accounts opened (24 months)
- `mort_acc`: Mortgage accounts
- `num_tl_op_past_12m`: Total accounts opened (12 months)
- `percent_bc_gt_75`: Bank cards >75% utilization
- `sub_grade`: Loan grade (A1-G5)
- `term`: Loan term (36 or 60 months)

The model returns both soft and hard predictions. In particular, the model returns the probability of default and also a final decision of whether the loan is charged off or fully paid and subsequently whether or not it should be rejected or accepted. 

## Deployment 

For deployment, the ideas from the notebook was modularised and written as python scripts. 

The final model was serialised and saved as pickle file. 
Docker was used for containerisation of the project and uv was used as the installation driver and package manager. 
FastAPI-based REST API was used to obtain the predictions of the models for the frontend which was hosted by Streamlit Cloud. Apart from that testing was also done on processing functions using pytest.

## Accessing the Front-End and Back-End 

1. Open the Streamlit app at `https://credit-risk-tool-g4qccubtezupxgueqw3y7k.streamlit.app/`
2. Open and start the API hosted by Render at `https://credit-risk-tool.onrender.com/` (This may take a minute or so for the back-end to wake up since I'm using the free tier). The interactive API documentation is also available using the above link.
2. Fill in the loan applicant details:
   - **Credit Profile**: Interest rate, FICO score, loan grade, term
   - **Credit History**: Inquiries, open accounts, account age
   - **Account Information**: Accounts opened, mortgage accounts
3. Click **"Get Prediction"** button (May take around 30-60 seconds if you are first loading it)
4. View the results including:
   - Default probability (0-100%)
   - Risk level classification (Low/Medium/High)
   - Model version
5. Review the input summary in the expandable section

### 🧪 API Testing

As mentioned before to access the interactive API documentation go to `https://credit-risk-tool.onrender.com/`

Example prediction request:
```bash
curl -X POST "https://credit-risk-tool.onrender.com/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "int_rate": 10.5,
      "fico_range_high": 750,
      "inq_last_6mths": 1,
      "open_il_12m": 2,
      "acc_open_past_24mths": 3,
      "mort_acc": 1,
      "num_tl_op_past_12m": 2,
      "percent_bc_gt_75": 25,
      "sub_grade": "A2",
      "term": "36 months"
    }]
  }'
```

## Technologies and Libraries Used 

Tech Stack used for Development: 
- Python
- Pandas 
- Numpy 
- Matplotlib, Seaborn 
- Sci-kit Learn 
- Feature Engine 
- MLFlow 
- Optuna 
- Docker 
- FastAPI 
- Streamlit 
- uv, poetry
- pytest

## Project Structure
```
├── app.py                 # Streamlit frontend application
├── src/
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # API entry point
│   │   ├── api.py        # API routes
│   │   ├── config.py     # API configuration
│   │   └── schemas/      # API data schemas
│   ├── predict.py        # Prediction logic
│   ├── train.py          # Model training
│   ├── pipeline.py       # ML pipeline
│   ├── config/           # Configuration files
│   ├── processing/       # Data processing modules
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
├── data/                 # Data files (raw, processed, interim)
├── model/                # Trained models
├── Dockerfile            # Docker image for API
├── Dockerfile.streamlit  # Docker image for frontend
├── docker-compose.yml    # Multi-container configuration
└── start.sh              # Startup script for local development
```
