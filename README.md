# Probability of Default (PD) Credit Risk Modelling

## Overview
This project focuses on building and evaluating models to estimate the **Probability of Default (PD)** for credit risk management. The Probability of Default is a key metric used by banks and financial institutions to assess the risk of borrowers failing to meet their debt obligations. In this notebook, we employ different machine learning techniques, such as logistic regression and gradient-boosted trees, to predict default probabilities based on provided features.

## Objective
The main goal of this project is to:
- Build and evaluate models that can predict the probability of default (PD).
- Demonstrate the use of logistic regression and gradient-boosted trees for credit risk modeling.
- Assess model performance and identify strategies for improving accuracy.

## Features
The notebook provides:
- Data preprocessing, handling missing values, and feature selection.
- Implementation of **logistic regression** and **gradient-boosted trees**.
- Model evaluation metrics, such as **accuracy, precision, recall**, and **ROC-AUC**.
- Visualization of model performance, ROC curves, and other analysis tools.

## Requirements
To run the notebook, ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- XGBoost (for gradient-boosted trees)

## How to Run
1. Clone or download the repository to your local machine.
2. Open the **Probability_of_Default_Credit_Risk_Modelling.ipynb** notebook in Jupyter or upload it to Google Colab.
3. Ensure all required libraries are installed in your environment.
4. Run each cell sequentially to train the models and evaluate their performance.

## Contents of the Notebook
- **Data Loading & Exploration**: Loads the dataset and provides a summary of the key variables.
- **Data Preprocessing**: Handles missing data, performs feature engineering, and splits the dataset into training and testing sets.
- **Modelling**:
    - Logistic regression model implementation.
    - Gradient-boosted trees for enhanced prediction performance.
- **Model Evaluation**:
    - Evaluation metrics including **accuracy, precision, recall**, and **ROC-AUC**.
    - ROC curves and other visual tools to compare model performance.
- **Improvements**: Suggestions on how to tune the model for better performance using hyperparameter tuning and other techniques.

## Key Learning Points
- Understanding how to model credit risk using machine learning.
- Importance of model evaluation metrics, especially when dealing with imbalanced datasets common in credit risk analysis.
- Learning how to improve model performance through feature engineering and hyperparameter tuning.

## Future Work
- Implement additional models, such as **Random Forest** or **Support Vector Machines**, to compare performance.
- Conduct more detailed hyperparameter tuning for optimal performance.
- Explore advanced techniques like **ensemble learning** or **deep learning** for improved accuracy.
