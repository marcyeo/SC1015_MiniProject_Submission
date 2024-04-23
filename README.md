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
      
- [Overall Ranking](): Overall ranking of our variables that affect loan status based on the machine learning models
        
                  
# Notebook Details
## Cleaning and Preparation
The dataset contained a mix of numerical and categorical columns. The data was cleaned to make it suitable for Exploratory Data Analysis and for procesing by our machine learning models. To do this, we:
1. Checked and removed missing values, duplicates and outliers within the dataset
2. Removed columns: 'Loan ID', 'Customer ID', 'Months since last delinquent', that are either used for customer identification or contained a large number of missing values
3. Standardised names of variables to make analysis easier, by removing underscores and standardising capitalisation

## Exploratory Data Analysis
1. Numerical Data

For our numerical data analysis, 

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

