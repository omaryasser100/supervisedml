# Supervised ML – Regression Techniques on the House Prices Dataset

This project applies a comprehensive set of supervised regression algorithms to the **House Prices dataset (extended version with 96 features)**.  
It demonstrates the full end-to-end machine learning workflow — from preprocessing and feature engineering to model comparison and evaluation — to predict housing prices accurately.

---

## Project Overview

The goal of this project is to predict the sale prices of houses using various regression models.  
It serves as a practical showcase of supervised machine learning workflows and model performance comparison on structured tabular data.

This repository covers:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering and scaling  
- Model training and hyperparameter tuning  
- Evaluation and comparison of multiple regression algorithms  

---

## Dataset

**Dataset:** House Prices – Advanced Regression Techniques (extended 96-feature version)  
- Each row represents a house with numerical and categorical features describing its characteristics.  
- The target variable is the **SalePrice**.  
- The dataset includes location, quality, size, and condition attributes.

**Features count:** 96 (after feature engineering and encoding)  
**Target:** Continuous variable – *SalePrice*  

---

## Workflow

1. **Data Loading and Inspection**  
   Load the dataset, inspect structure, handle missing values, and encode categorical features.  

2. **Preprocessing and Feature Engineering**  
   - Imputation of missing values  
   - Label encoding or one-hot encoding  
   - Feature scaling (StandardScaler or MinMaxScaler)  
   - Outlier detection and removal  

3. **Exploratory Data Analysis (EDA)**  
   - Correlation heatmaps  
   - Feature distributions  
   - Target variable skewness  
   - Top correlated predictors  

4. **Model Training**  
   Trained and compared multiple regression models, including:
   - Linear Regression  
   - Ridge Regression  
   - Lasso Regression  
   - ElasticNet  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  
   - XGBoost  
   - LightGBM  
   - CatBoost  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Regressor (SVR)  

5. **Hyperparameter Optimization**  
   - Used GridSearchCV or RandomizedSearchCV  
   - Cross-validation (K-Fold)  

6. **Model Evaluation Metrics**
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score  

7. **Model Comparison**
   - Created a leaderboard comparing all trained models  
   - Highlighted trade-offs between bias, variance, and computational cost  

## Repository Structure

