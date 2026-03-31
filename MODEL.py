import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ===== 0. Configuration =====
# Path to your CSV file
csv_path = 'melb_data.csv' 

# Name of the target column (the price column)
target_col = 'Price' 

# Feature columns to use ('auto' to detect numeric columns)
feature_columns = 'auto'

print("Configuration set!")
print(f"CSV Path: {csv_path}")
print(f"Target Column: {target_col}")

# ===== 1. Load and Explore Data =====
try:
    home_data = pd.read_csv(csv_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File not found at {csv_path}")
    raise

print(f"\nDataset shape: {home_data.shape}")
print(f"\nColumn names: {home_data.columns.tolist()}")

# ===== 2. Prepare Features and Target =====
if target_col not in home_data.columns:
    raise KeyError(f"Column '{target_col}' does not exist.")

y = home_data[target_col]

if feature_columns == 'auto':
    feature_columns = [col for col in home_data.columns 
                       if home_data[col].dtype in ['int64', 'float64'] 
                       and col != target_col]

X = home_data[feature_columns].copy()

# ===== 3. Handle Missing Values =====
missing_count = X.isnull().sum()
if missing_count.sum() > 0:
    print("\nDropping rows with missing values...")
    X = X.dropna()
    y = y.loc[X.index]

# Remove any rows with missing target values
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]
print(f"✅ Final dataset shape: {X.shape}")

# ===== 4. Split Data =====
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, 
                                                    test_size=0.2, random_state=1)

# ===== 5. Model 1: Decision Tree (Baseline) =====
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
dt_predictions = dt_model.predict(val_X)
dt_mae = mean_absolute_error(val_y, dt_predictions)
print(f"Decision Tree - Validation MAE: ${dt_mae:,.0f}")

# ===== 6. Hyperparameter Tuning (Tree Size) =====
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {}
for leaf_size in candidate_max_leaf_nodes:
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_size, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    scores[leaf_size] = mae

best_tree_size = min(scores, key=scores.get)
print(f"Best max_leaf_nodes: {best_tree_size} with MAE: ${scores[best_tree_size]:,.0f}")

# ===== 7. Model 2: Random Forest =====
rf_model = RandomForestRegressor(random_state=1, n_estimators=100)
rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_predictions)
print(f"Random Forest - Validation MAE: ${rf_mae:,.0f}")

# ===== 8. Summary =====
print("=" * 60)
print("MACHINE LEARNING PROJECT SUMMARY")
print("=" * 60)
print(f"Best Model: Random Forest")
print(f"Best Validation MAE: ${rf_mae:,.0f}")

# Feature Importance
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_columns, 
                                   'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\nTop 3 Most Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f" {row['Feature']}: {row['Importance']:.4f}")
print("=" * 60)
