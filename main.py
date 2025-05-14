# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Optional advanced model
from sklearn.metrics import mean_squared_error, r2_score
from load_data import csv_to_df
# %%
df_test = csv_to_df("home-data-for-ml-course/test.csv")
df_train = csv_to_df("home-data-for-ml-course/train.csv")

df_test.head(), df_train.head()
# %%

df_train.info()
# %%
df_train.describe()
# %%
df_train.head()
# %%
#Missing values in each column
missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)
missing_values = pd.DataFrame(missing_values)
missing_values.reset_index(inplace=True)
missing_values.columns = ['Column Name', 'Missing Values']
print(missing_values)
# %%
train_features = ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
target = "SalePrice"
df_train[train_features].head()
# %%
#Replace missing values with mean
df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean(), inplace=True)
df_train["LotArea"].fillna(df_train["LotArea"].mean(), inplace=True)
df_train["OverallQual"].fillna(df_train["OverallQual"].mean(), inplace=True)
df_train["OverallCond"].fillna(df_train["OverallCond"].mean(), inplace=True)
#Replacing missing values with mean in test set
df_test["LotFrontage"].fillna(df_test["LotFrontage"].mean(), inplace=True)
df_test["LotArea"].fillna(df_test["LotArea"].mean(), inplace=True)
df_test["OverallQual"].fillna(df_test["OverallQual"].mean(), inplace=True)
df_test["OverallCond"].fillna(df_test["OverallCond"].mean(), inplace=True)

#Droping with missing price values
df_train.dropna(subset=["SalePrice"], inplace=True)
df_train[train_features].isnull().sum()
# %%
#Plotting the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x="LotFrontage", y="SalePrice", data=df_train)
plt.title("LotFrontage vs SalePrice")
plt.xlabel("LotFrontage")
plt.ylabel("SalePrice")
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x="LotArea", y="SalePrice", data=df_train)
plt.title("LotArea vs SalePrice")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x="OverallQual", y="SalePrice", data=df_train)
plt.title("OverallQual vs SalePrice")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x="OverallCond", y="SalePrice", data=df_train)
plt.title("OverallCond vs SalePrice")
plt.xlabel("OverallCond")
plt.ylabel("SalePrice")
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df_train[train_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
# %%
#Scaling the data
X_train = df_train[train_features]
y_train = df_train[target]

X_test = df_test[train_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
# %%
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# %%
y_pred = model.predict(X_test_scaled)
# %%
from sklearn.model_selection import train_test_split

X_train_full = df_train[train_features]
y_train_full = df_train[target]

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
model = LinearRegression()
model.fit(X_train_scaled, y_train_split)
y_val_pred = model.predict(X_val_scaled)
mse = mean_squared_error(y_val_split, y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_split, y_val_pred)

print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val_split, y=y_val_pred)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.plot([y_val_split.min(), y_val_split.max()], [y_val_split.min(), y_val_split.max()], 'r--')
plt.show()
# %%