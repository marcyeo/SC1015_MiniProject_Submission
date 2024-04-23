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
    * [categorical_credit](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/datasets/categorical_credit.csv): Contains only cleaned categorical columns
    * [cleaned_credit](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/datasets/cleaned_credit.csv): Dataset after cleaning
    * [combined](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/datasets/combined.csv): Combined Dataset of cleaned numerical and categorical columns 
    * [credit_train](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/datasets/credit_train.csv): Original Dataset
    * [numerical_credit](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/datasets/numerical_credit.csv): Contains only cleaned numerical columns
      
- [Overall Ranking](https://github.com/marcyeo/SC1015_MiniProject_Submission/blob/main/Overall_Ranking.xlsx): Overall ranking of our variables that affect loan status based on the machine learning models

Note: Kindly download all excel files for viewing 
                  
# Notebook Details
## Cleaning and Preparation
The dataset contained a mix of numerical and categorical columns. The data was cleaned to make it suitable for Exploratory Data Analysis and for procesing by our machine learning models. To do this, we:
1. Checked and removed missing values, duplicates and outliers within the dataset
2. Removed columns: 'Loan ID', 'Customer ID', 'Months since last delinquent', that are either used for customer identification or contained a large number of missing values
3. Standardised names of variables to make analysis easier, by removing underscores and standardising capitalisation

## Exploratory Data Analysis
We split our numerical and categorical columns into distinct columns to analyse their impact on loan status.

1. Numerical Data

- For our numerical data analysis, we grouped the columns ('Loan Status','Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Years of Credit History', 'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens') into a numerical dataframe
- Reclassified 'Number of Credit Problems', 'Bankruptcies' and 'Tax Liens' into categorical variables as a large proportion of their values could be split into distinct categories.
- Converted loan status into a dummy variable for comparison with our numerical data.
- Violin Plots to analyse distribution of each numerical variable to identify trends
- Correlation table and matrix for each variable with loan default status
- Used plotly to visualise each numerical variable with loan status, to observe trends between values that yield a loan status value of 0 or 1
- 'Current Loan Amount', 'Credit Score', 'Annual Income' and 'Maximum Open Credit' were the most strongly correlated with loan status based on our EDA


2. Categorical Data

## Machine Learning
1. Logistic Regression
- Converted categorical variables into dummy variables
- Trained and tested the logistic regression model
- Printed the model's results ranking of variables that determine loan status
- Checked accuracy metrics

2. Random Forest

3. Multilayer Perceptron Neural Network

4. Neural Network for SHAP Analysis



## Conclusion
Using Excel (File:Overall_Ranking) to collate the ranking of our 3 machine learning models and calculating every variable's average ranking score, we derived our top 10 most significant variables determing loan status.

Based on these variables, we propose a scoring system that banks can use to rate the risk level of loan default for every customer profile. The 10 variables will serve as components to be graded, with the results from the SHAP values used to determine the scoring. The scores would aid banks in customer profiling, which banks can then amend loan disbursement policies to cater to different levels of loan default risk, reducing the risk of customer loan default.

With the complex nature of loan defaults, further developments of our project could analyse the interactions between individual variables using neural networks and how these interactions would influence loan status using techniques like Partial Dependence Plots (PDPs) or Individual Conditional Expectation (ICE). Clustering of customers using k-means clustering based on different sets of variables could be used to more accurately identify the range and combination of variable values that would group these customers under different risk levels for loan default.

## What we learnt from the Project
- Well defined problem formulation to derive data driven insights
- Data Visualisation techniques using plotly and hvplot library
- New machine learning models such as Logistic Regression from sklearn and Neural Networks
- Collaborating using GitHub
- Classification Concepts such as F1 Score
- SHAP graph analysis for interpreting Neural Network

## References 
1. https://acerta.ai/blog/understanding-machine-learning-with-shap-analysis/
2. https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/

