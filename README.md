# SC1015_MiniProject

## About
Introducing our SC1015 mini-project, where we performed analysis on the [Credit_card dataset from Kaggle](https://www.kaggle.com/datasets/fatmayousufmohamed/credit-card/data). Please view the source code in the following order:
1. [Data Cleaning and preparation](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/data%20cleaning.ipynb)
2. [Exploratory Data Analysis](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/exploratory%20data%20analysis.ipynb)
3. Machine learning: [Logistic Regression](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/Logistic%20Regression.ipynb), [Random Forest](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/randomforest.ipynb), [Multilayer Perceptron Neural Network](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/multilayer%20perceptron%20nn.ipynb), [Neural Network for SHAP Analysis](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/keras%20nn.ipynb)

## Problem Definition
- What are the top factors that influence loan default?
- What can banks do to reduce risk of loan defaults?

## Contributors (FDDB):
- Yoong Hong Jun, Nicholas - Data Cleaning, Data Visualiation
- Marcus Yeo Xian Sheng - Logistic Regression, Random Forest
- Shermaine Yau Yu Shuen - Neural Networks

## Other Files Included
- [Excel Datasets](https://github.com/marcyeo/SC1015_MiniProject_Submission/tree/main/datasets)
    * [categorical_credit](): Contains only cleaned categorical columns
    * [cleaned_credit](): Dataset after cleaning
    * [combined](): Combined Dataset of cleaned numerical and categorical columns 
    * [credit_train](): Original Dataset
    * [numerical_credit](): Contains only cleaned numerical columns
      
- [Overall Ranking]()
        
                  
# Notebook Details
## Cleaning and Preparation
The dataset contained a mix of numerical and categorical columns. The data was cleaned to make it suitable for Exploratory Data Analysis and for procesing by our machine learning models. To do this, we:
1. Checked and removed missing values, duplicates and outliers within the dataset
2. Removed columns: 'Loan ID', 'Customer ID', 'Months since last delinquent', that are either used for customer identification or contained a large number of missing values
3. Standardised names of variables to make analysis easier

## Exploratory Data Analysis
1. Numerical Data
2. Categorical Data

## Machine Learning
1. Logistic Regression

2. Random Forest

3. Multilayer Perceptron Neural Network

4. Neural Network for SHAP Analysis



## Conclusion

## What we learnt from the Project
- Well defined problem formulation to derive data driven insights
- Data Visualisation techniques using plotly and hvplot library
- New machine learning models such as Logistic Regression from sklearn and Neural Networks
- Collaborating using GitHub
- Classification Concepts such as F1 Score
- SHAP graph analysis for interpreting Neural Network

## References 


