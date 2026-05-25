# 🏠 California Housing Price Prediction (Linear Regression Project)

## 📌 Project Overview
This project applies machine learning techniques to predict median house prices in California using the **California Housing dataset**. The goal is to build a regression model and analyze how different features influence house prices.

---

## 📊 Dataset Information
- Source: sklearn.datasets.fetch_california_housing
- Records: 20,640 rows
- Features:
  - MedInc (Median Income)
  - HouseAge
  - AveRooms
  - AveBedrms
  - Population
  - AveOccup
  - Latitude
  - Longitude
- Target:
  - MedHouseVal (Median House Value)

---

## 🔍 Exploratory Data Analysis (EDA)
- Checked missing values (none found)
- Checked duplicates (none found)
- Summary statistics using `.describe()`
- Correlation heatmap
- Feature distribution plots
- Boxplots to detect outliers
- Geographical visualization of house prices

---

## 🤖 Machine Learning Models Used

### 1. Linear Regression
- Baseline model
- R² Score: ~0.57

### 2. Decision Tree Regressor
- R² Score: ~0.62

### 3. Random Forest Regressor
- Best performing model
- R² Score: ~0.80+

### 4. Gradient Boosting Regressor
- R² Score: ~0.77

---

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---
