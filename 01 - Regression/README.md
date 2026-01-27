## House Price Prediction using Regression Models

## Project Overview

This project focuses on predicting house prices using multiple regression algorithms and comparing their performance to identify the most suitable model. The goal is to understand how different regression techniques behave on the same dataset and why model selection matters in Machine Learning.

The project is designed as a learning + portfolio project, following an end-to-end ML workflow used in real-world industry scenarios.

## Problem Statement

Given housing-related features such as median income, house age, location, and room information, predict the median house value.

This is a supervised regression problem.

## Dataset Description

Dataset: California Housing Dataset

Rows: ~20,000

Features: 9 numerical features

Target variable: MedHouseVal (Median House Value)

### Key Features:

MedInc – Median income

HouseAge – Average house age

AveRooms – Average number of rooms

Latitude, Longitude – Location details

Population, AveOccup – Demographic features

### Project Workflow

Each regression model follows the same ML pipeline:

Dataset Loading & Understanding

Exploratory Data Analysis (EDA)

Feature & Target Separation

Feature Scaling

Model Training

Prediction

Model Evaluation

Visualization (Actual vs Predicted)

EDA is performed once to understand the data. The same cleaned dataset is reused across all models to ensure a fair comparison.

## Models Implemented

All models are implemented using the same dataset to compare performance effectively.

### 1️⃣ Linear Regression

Baseline model

Assumes linear relationship between features and target

### 2️⃣ Polynomial Regression

Captures non-linear relationships

Uses polynomial feature transformation

### 3️⃣ Ridge Regression

Linear model with L2 regularization

Helps reduce overfitting

### 4️⃣ Lasso Regression

Linear model with L1 regularization

Performs feature selection

### 5️⃣ Elastic Net Regression

Combination of Ridge and Lasso

Useful when features are correlated

### Evaluation Metrics

All models are evaluated using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² Score

## Model Comparison

Model	                   MAE	 RMSE	 R²Score

Linear Regression	       ~0.50	~0.68	~0.64

Polynomial Regression	   ~0.42	~0.60	~0.72

Ridge Regression	       ~0.50	~0.68	~0.64

Lasso Regression	       ~0.50	~0.68	~0.64

Elastic Net Regression	 ~0.56	~0.75	~0.56

## Final Conclusion

Polynomial Regression performs the best among all models.

House prices show non-linear relationships with income, location, and room features.

Regularization models help control overfitting but do not outperform Polynomial Regression for this dataset.

## Best Model: Polynomial Regression
