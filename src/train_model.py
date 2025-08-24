# train_model.py
import json
import joblib
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
from preprocess import load_data, preprocess_data  # your preprocessing functions

# ------------------------
# 1. Load Data
# ------------------------

# Assuming the data is in a CSV file
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "melb_data.csv"
df = load_data(data_path)
X, y, preprocessor = preprocess_data(df)

# ------------------------
# 2. Load best model info from JSON
# ------------------------
json_path = "models/best_model_results.json"
with open(json_path, "r") as f:
    best_model_info = json.load(f)

best_model_name = best_model_info["best_model_name"]
best_params = best_model_info["best_params"]

print(f"Training final model: {best_model_name} with params {best_params}")

# ------------------------
# 3. Map model name to estimator
# ------------------------
model_mapping = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "LinearRegression": LinearRegression()
}

model = model_mapping[best_model_name]

# ------------------------
# 4. Create full pipeline
# ------------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ------------------------
# 5. Set hyperparameters from JSON
# ------------------------
clf.set_params(**best_params)

# ------------------------
# 6. Train on full dataset
# ------------------------
clf.fit(X, y)

# ------------------------
# 7. Save the trained model
# ------------------------
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "final_model.pkl"
joblib.dump(clf, model_path)
print(f"Trained model saved to {model_path}")
