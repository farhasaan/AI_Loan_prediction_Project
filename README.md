AI Loan Approval Prediction

A Machine Learning project to predict loan approval decisions based on applicant financial and demographic information.

1. Project Overview

This project aims to develop a machine learning model that predicts whether a loan application is likely to be approved or rejected.

The model analyzes applicant information, such as annual income, loan amount, credit score, employment status, and asset values, to generate a prediction.

This project covers the machine learning development workflow, from data preparation and model experimentation to model evaluation and deployment through a Streamlit application.

2. Business Problem

Financial institutions need to evaluate loan applications efficiently while managing credit risk.

Manual evaluation can require significant time and resources. Machine learning can help identify patterns in historical loan application data and provide decision-support predictions.

Business objective:

Support faster loan application assessments.

Identify patterns associated with loan approval.

Provide data-driven insights for preliminary credit evaluation.

3. Project Objectives

Perform exploratory data analysis (EDA).

Prepare and preprocess the dataset.

Train and compare multiple machine learning models.

Evaluate model performance using classification metrics.

Track experiments using MLflow.

Save the trained model for inference.

Build an interactive prediction application using Streamlit.

4. Dataset

Source: Kaggle – Loan Approval Prediction Dataset

The dataset contains applicant financial information and historical loan approval decisions.

Features

Feature

Description

no_of_dependents

Number of dependents

education

Education status

self_employed

Self-employment status

income_annum

Annual income

loan_amount

Requested loan amount

loan_term

Loan term

cibil_score

Credit score

residential_assets_value

Residential asset value

commercial_assets_value

Commercial asset value

luxury_assets_value

Luxury asset value

bank_asset_value

Bank asset value

Target variable: loan_status

0 = Rejected

1 = Approved

The loan_id column is not used as a predictive feature.

5. Machine Learning Workflow

Data collection

Data cleaning and preprocessing

Exploratory data analysis

Feature engineering

Model training and comparison

Hyperparameter tuning

Model evaluation

Experiment tracking with MLflow

Model serialization

Prediction through Streamlit

6. Model Evaluation

Model performance is evaluated using classification metrics, including:

Accuracy

Precision

Recall

F1-score

ROC-AUC

PR-AUC

Best model: To be completed based on experiment results.

Best model performance: To be completed using actual evaluation results.

7. Experiment Tracking

MLflow is used to track machine learning experiments, including model parameters, evaluation metrics, and experiment results.

Experiment name:

loan-approval-prediction

Tracking backend:

SQLite

8. Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

MLflow

Joblib

Streamlit

Jupyter Notebook

Visual Studio Code

9. Project Structure

AI_Loan_prediction_Project/
│
├── app.py
├── Experiment.ipynb
├── .gitattributes
├── artifacts/
│   ├── loan_approval_model.joblib
│   ├── metrics.json
│   ├── model_metadata.json
│   └── ...
│
├── data/                
├── mlruns/
└── README.md

10. Run the Application

Clone the repository:

git clone https://github.com/farhasaan/AI_Loan_prediction_Project.git

Navigate to the project directory:

cd AI_Loan_prediction_Project

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

Note: The trained model artifact and required dependencies must be available locally before running the application. The dataset is not included in the repository.

11. Limitations

The model learns from historical data and may not represent all real-world lending conditions.

Predictions depend on the quality and representativeness of the training dataset.

Model predictions should not be treated as the sole basis for actual lending decisions.

Further validation is required before real-world financial deployment.

12. Future Improvements

Improve model performance through additional experimentation.

Implement model monitoring and performance tracking.

Deploy the application to a cloud platform.

Improve explainability using model interpretation techniques.

Evaluate fairness and potential bias in loan approval predictions.

Disclaimer: This project is developed for educational and portfolio purposes. It is not intended to replace professional credit risk assessment or actual financial decision-making.