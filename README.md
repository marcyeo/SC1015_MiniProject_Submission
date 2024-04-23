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
Our main objective was to analyse how the different variables in the dataset were correlated with 'Loan Status'. 
The dataset contained a mix of numerical and categorical columns. The data was cleaned to make it suitable for Exploratory Data Analysis and for procesing by our machine learning models. To do this, we:
1. Checked and removed missing values, duplicates and outliers within the dataset
2. Removed columns: 'Loan ID', 'Customer ID', 'Months since last delinquent', that are either used for customer identification or contained a large number of missing values
3. Standardised names of variables to make analysis easier, by removing underscores and standardising capitalisation

## Exploratory Data Analysis
We split our numerical and categorical columns into distinct columns to analyse their impact on loan status.

1. Numerical Data

- For our numerical data analysis, we grouped the columns ('Loan Status','Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Years of Credit History', 'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens') into a numerical dataframe
- Further data cleaning: Reclassified 'Number of Credit Problems', 'Bankruptcies' and 'Tax Liens' into categorical variables as a large proportion of their values could be split into distinct categories.
- Converted loan status into a dummy variable for comparison with our numerical data.
- Violin Plots to analyse distribution of each numerical variable to identify trends
- Correlation table and matrix for each variable with loan default status
- Used plotly to visualise each numerical variable with loan status, to observe trends between values that yield a loan status value of 0 or 1
- 'Current Loan Amount', 'Credit Score', 'Annual Income' and 'Maximum Open Credit' were the most strongly correlated with loan status based on our EDA


2. Categorical Data
- For our categorical data analysis, we grouped the columns('Loan Status', 'Term', 'Years in current job', 'Home Ownership', 'Purpose', 'Number of Credit Problems', 'Bankruptcies', 'Tax Liens') into a categorical dataframe
- Reclassified numerical variables ('Number of Credit Problems', 'Bankruptcies', 'Tax Liens') into categorical: Assigned categories '0' and '>=1'
- Visualised categorical variables with respect to loan status using grouped barplots to analyse correlation with loan status of 'Charged Off' and 'Fully Paid'
- 'Term', 'Years in current job', 'Home Ownership' and 'Purpose' had greatest deviations between the categories with respect to loan status, and were likely to be stronger predictors based on our EDA

## Machine Learning
1. Logistic Regression
- Converted categorical variables into dummy variables
- Trained and tested the logistic regression model
- Printed the model's ranking of variables that determine loan status
- Checked accuracy metrics (train vs test performance, classification report)

2. Random Forest
- One-hot encoded categorical variables
- Trained and tested Random Forest Classifier model (Model #1)
- Checked accuracy metrics (train vs test performance, classification report) for Model #1
- Printed Model #1's ranking of variables that determine loan status
- Tuned hyperparameters using Randomized Search, finding best estimator (Model #2)
- Plotted Random Forest Tree for Model #2
- Checked accuracy metrics (train vs test performance, classification report) for Model #2
- Printed Model #2's ranking of variables that determine loan status

3. Multilayer Perceptron Neural Network
- Removed mean and scaled each numerical variable to unit variance
- One-hot encoded categorical variables
- Trained and tested Multilayer Perceptron model
- Printed model's ranking of variables that determine loan status
- Checked accuracy metrics (train vs test performance, classification report

4. Neural Network for SHAP Analysis
- Removed mean and scaled each numerical variable to unit variance
- One-hot encoded categorical variables
- Created a neural network model with input layer (ReLU activation), 2 hidden layers (ReLU activation), and output layer (Sigmoid activation)
- Trained and tested Neural Network model, printing training process with accuracy, loss, precision, and recall for each Epoch
- Printed weights for each layer
- Visualised training process with Line graphs:  Loss, Accuracy, Precision and Recall History
- Checked accuracy metrics (train vs test performance, classification report)
- SHapley Additive exPlanations analysis to analyse importance of variables in predicting Loan Status and correlation with Loan Status

## Conclusion
Using Excel (File:Overall_Ranking) to collate the ranking of our 3 machine learning models and calculating every variable's average ranking score, we derived our top 10 most significant variables determing loan status.

![image](https://github.com/marcyeo/SC1015_MiniProject_Submission/assets/147054465/2f5c708c-3858-4fdb-9cb3-661f9e3d81ff)

Note: 'Home Ownership' and 'Term' are categorical variables.

Based on these variables, we propose a scoring system that banks can use to rate the risk level of loan default for every customer profile. The 10 variables will serve as components to be graded, with the results from the SHAP values used to determine the scoring. The scores would aid banks in customer profiling, which banks can then amend loan disbursement policies to cater to different levels of loan default risk, reducing the risk of customer loan default.

With the complex nature of loan defaults, further developments of our project could analyse the interactions between individual variables using neural networks and how these interactions would influence loan status using techniques like Partial Dependence Plots (PDPs) or Individual Conditional Expectation (ICE). Clustering of customers using k-means clustering based on different sets of variables could be used to more accurately identify the range and combination of variable values that would group these customers under different risk levels for loan default.

## What we learnt from the Project
- Well defined problem formulation to derive data driven insights
- Data Visualisation techniques using plotly and hvplot library
- New machine learning models such as Random Forest and Logistic Regression from sklearn, and Neural Networks (sklearn and keras library)
- Collaborating using GitHub
- Classification Concepts such as F1 Score
- SHAP graph analysis for interpreting Neural Network

## References 
1. https://www.kaggle.com/datasets/fatmayousufmohamed/credit-card/data
2. https://acerta.ai/blog/understanding-machine-learning-with-shap-analysis/
3. https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/
4. https://www.cnbc.com/2024/02/06/credit-card-delinquencies-surged-in-2023-indicating-financial-stress-new-york-fed-says.html
5. https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141
6. https://machinelearningmastery.com/neural-networks-crash-course/
7. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
8. https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
9. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
10. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
11. https://scikit-learn.org/stable/modules/neural_networks_supervised.htmlhttps://datascientest.com/en/shap-what-is-it#:~:text=SHapley%20Additive%20exPlanations%2C%20more%20commonly,each%20feature%20or%20feature%20value.
12. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
13. https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier
14. https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
