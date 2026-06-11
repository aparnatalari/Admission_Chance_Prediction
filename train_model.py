import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("dataset/Admission_Predict.csv")

# Remove Serial Number column
data = data.drop("Serial No.", axis=1)

# Features and Target
X = data.drop("Chance of Admit ", axis=1)
y = data["Chance of Admit "]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/admission_model.pkl")

print("Model trained successfully!")