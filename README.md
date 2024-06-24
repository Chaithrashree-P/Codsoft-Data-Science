# Codsoft-Data-Science


# Titanic Survival Prediction

## Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the RMS Titanic sank after colliding with an iceberg, resulting in the deaths of 1502 out of 2224 passengers and crew. This project aims to build a predictive model to determine the likelihood of a passenger surviving the disaster based on various features such as age, gender, socio-economic class, and more.

## Dataset

The dataset used for this project contains information about individual passengers on the Titanic. It includes the following columns:

- **Survived**: Survival (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Objective

1. **Data Understanding and Cleanup**: Load the dataset and perform necessary data cleaning, such as handling missing values and encoding categorical variables.
2. **Model Building**: Build several classification models to predict passenger survival.
3. **Model Evaluation**: Compare the performance of different models using appropriate evaluation metrics.
4. **Hyperparameter Tuning**: Fine-tune the hyperparameters of the best-performing model.
5. **Final Evaluation**: Evaluate the final model and interpret the results.

## Steps Involved

### 1. Data Preprocessing

- **Handling Missing Values**: Fill missing values for 'Age' with the median and 'Embarked' with the mode. Drop the 'Cabin' column due to a high number of missing values.
- **Feature Selection**: Drop irrelevant columns like 'Name' and 'Ticket'.
- **Encoding**: Encode categorical variables such as 'Sex' and 'Embarked'.

### 2. Data Splitting

- Split the dataset into training and testing sets to evaluate model performance.

### 3. Model Building

- Train several classification models including Logistic Regression, Random Forest, and Support Vector Machine (SVM).

### 4. Model Evaluation

- Evaluate each model using metrics such as accuracy, precision, recall, and the confusion matrix.
- Visualize the performance using heatmaps and classification reports.

### 5. Hyperparameter Tuning

- Perform hyperparameter tuning on the best-performing model (Random Forest in this case) using GridSearchCV.

### 6. Final Model Evaluation

- Evaluate the final model with the best hyperparameters and interpret the results.
- Display the best hyperparameters and conclude the model training and evaluation process.

## Conclusion

This project provides a comprehensive approach to building and evaluating machine learning models for binary classification problems. The Titanic dataset serves as a classic example, highlighting the importance of data preprocessing, model selection, evaluation, and tuning in the machine learning workflow.



# Credit Card Fraud Detection

## Overview

Credit card fraud detection is a critical problem in the banking industry. This project focuses on building predictive models to detect fraudulent transactions using machine learning techniques. The dataset contains anonymized credit card transactions labeled as fraudulent or genuine.

## Dataset

The dataset used in this project is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It includes the following relevant columns:

- **Time**: Time elapsed between each transaction and the first transaction in seconds.
- **Amount**: Transaction amount.
- **Class**: Target variable indicating fraud (1) or genuine transaction (0).

## Objective

1. **Data Preprocessing**: Load the dataset, handle missing values, normalize numerical features, and split the dataset into training and testing sets.
2. **Handling Class Imbalance**: Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the training set.
3. **Model Building**: Build and evaluate machine learning models:
   - RandomForestClassifier
   - LogisticRegression
   - DecisionTreeClassifier (added later)
4. **Model Evaluation**: Evaluate each model using classification metrics such as precision, recall, and precision-recall AUC.
5. **Optional Model**: Include DecisionTreeClassifier in the model selection process.

## Steps Involved

### 1. Data Preprocessing

- Load the dataset and check basic information.
- Handle missing values (if any) and normalize the 'Amount' feature using StandardScaler.
- Drop the 'Time' column as it may not be useful for modeling.

### 2. Handling Class Imbalance

- Use SMOTE to oversample the minority class (fraudulent transactions) in the training set to improve model performance.

### 3. Model Building and Evaluation

- Train and evaluate the RandomForestClassifier and LogisticRegression models on the balanced training set.
- Calculate and print classification reports, precision-recall AUC, and confusion matrix for model evaluation.

### 4. Optional Model - DecisionTreeClassifier

- Include DecisionTreeClassifier as an additional model to compare performance with RandomForestClassifier and LogisticRegression.
- Train the DecisionTreeClassifier model on the original training set without SMOTE oversampling.
- Evaluate the model using classification metrics similar to other models.

## Conclusion

This project demonstrates the application of machine learning techniques for credit card fraud detection, emphasizing data preprocessing, handling class imbalance, model building, and evaluation. The inclusion of multiple models allows for comparison and selection of the most suitable approach for detecting fraudulent transactions.

## Libraries Used

- pandas: Data manipulation and analysis.
- numpy: Scientific computing library.
- scikit-learn: Machine learning library for model building, evaluation, and preprocessing.
- imbalanced-learn (imblearn): Library for handling class imbalance using techniques like SMOTE.



# Iris Flower Classification

This project involves classifying Iris flowers into three species: setosa, versicolor, and virginica based on their sepal and petal measurements. The Iris dataset, which is widely used for introductory classification tasks, is used to train and evaluate the model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Iris flower dataset is a classic dataset for pattern recognition. It contains 150 observations of iris flowers with four features: sepal length, sepal width, petal length, and petal width. The objective is to classify the iris flowers into three species based on these measurements.

## Dataset

The dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica). There are four features measured from each sample: the lengths and the widths of the sepals and petals, in centimeters.

## Installation

To run this project, you will need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using pip:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn

