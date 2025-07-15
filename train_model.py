import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import numpy as np

# Load dataset
df = pd.read_csv("housing.csv")

# Drop the categorical column
df = df.drop('ocean_proximity', axis=1)

# Handle missing values
imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Add combined feature
df_imputed["rooms_per_household"] = df_imputed["total_rooms"] / df_imputed["households"]

# Scale features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

# Features and target
X = df_scaled.drop("median_house_value", axis=1)
y = df_scaled["median_house_value"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    rose = np.sqrt(mean_squared_error(y_test, predict))  # Fixed
    mae = mean_absolute_error(y_test, predict)
    print(f"{name} => ROSE: {rose:.4f}, MAE: {mae:.4f}")


# Save best model (Random Forest) using pickle
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

# Save to same directory as this script
output_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
with open(output_path, "wb") as file:
    pickle.dump(best_model, file)

print("✅ Model training complete.")
print(f"📁 Model saved at: {output_path}")
print("📦 File exists:", os.path.exists(output_path))
