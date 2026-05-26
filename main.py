import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load


# ----------------------------
# 1. Load Dataset
# ----------------------------
data = fetch_california_housing(as_frame=True)

df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1)

print("Dataset Shape:", df.shape)
print(df.head())


# ----------------------------
# 2. Basic Data Check
# ----------------------------
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nStatistics:\n", df.describe())


# ----------------------------
# 3. Split Data
# ----------------------------
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# 4. Linear Regression Model
# ----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))


# ----------------------------
# 5. Random Forest Model
# ----------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print("R2 Score:", r2_score(y_test, rf_pred))


# ----------------------------
# 6. Decision Tree Model
# ----------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("R2 Score:", r2_score(y_test, dt_pred))


# ----------------------------
# 7. Gradient Boosting Model
# ----------------------------
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("\n--- Gradient Boosting ---")
print("R2 Score:", r2_score(y_test, gb_pred))


# ----------------------------
# 8. Feature Importance (Random Forest)
# ----------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)


# ----------------------------
# 9. Save Model
# ----------------------------
dump(rf, "model.joblib")
print("\nModel saved as model.joblib")


# ----------------------------
# 10. Load Model (Test)
# ----------------------------
loaded_model = load("model.joblib")
test_pred = loaded_model.predict(X_test)


# ----------------------------
# 11. Visualization
# ----------------------------

# Actual vs Predicted
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


# Feature Importance Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance (Random Forest)")
plt.show()


# Distribution of target
sns.histplot(df["MedHouseVal"], kde=True)
plt.title("House Price Distribution")
plt.show()


# Residual Plot
residuals = y_test - rf_pred
sns.scatterplot(x=rf_pred, y=residuals)
plt.axhline(0, color="red")
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
