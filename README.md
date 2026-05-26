California Housing Price Prediction:

Project Overview

This project builds and compares multiple Machine Learning models to predict median house prices in California using the scikit-learn California Housing dataset. It includes data analysis, visualization, feature engineering insights, and model evaluation.

Dataset
Source: sklearn.datasets.fetch_california_housing
Rows: 20,640
Features: 8 numerical features
Target: MedHouseVal (Median House Value)


⚙️ Tech Stack
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn


Exploratory Data Analysis (EDA):

The dataset was explored to understand structure, relationships, and data quality.

✔ Steps Performed:
Checked dataset shape and structure
Verified missing values (none found)
Checked duplicate values (none found)
Used describe() for statistical summary
Visualized:
Distribution of house prices
Correlation heatmap
Boxplots for feature spread
Scatter plots (Income vs Price, Location vs Price)


Model Building & Training :

Multiple regression models were trained and compared.

✔ Steps Performed:
Split dataset into training (80%) and testing (20%)
Trained models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor


📈 Model Evaluation Metrics:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² Score


🏆 Model Performance
Model	R² Score
Linear Regression	~0.57
Decision Tree	~0.62
Gradient Boosting	~0.77
Random Forest	~0.80 (Best Model)


📌 Conclusion
Random Forest performed best among all models
Income is the most important factor affecting house prices
Location plays a major role in pricing trends

