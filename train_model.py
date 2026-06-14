import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_maintenance_model(data_path="data/machine_data.xlsx", model_path="models/maintenance_model.pkl"):
    """
    Trains a RandomForest model to predict machine breakdowns.
    Includes robust error handling and directory creation.
    """
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist. Cannot train model.")
        return

    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    features = [
        'Progressive Work Hours',
        'Progressive Breakdown Hrs',
        'Progressive Maintenance Hrs',
        'Actual Progressive % Availability',
        'Actual Progressive % Utilization',
        'Total Running Hrs.',
        'Life Completed (in Yrs)'
    ]
    
    target = 'Working Status'

    print("Checking dataset validity...")
    # Safe fallback if columns are missing
    for col in features:
        if col not in df.columns:
            print(f"Warning: Missing column '{col}'. Filling with 0.")
            df[col] = 0
            
    if target not in df.columns:
        print(f"Error: Target column '{target}' is missing. Cannot train model.")
        return

    print("Cleaning and preparing data...")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Clean target
    df = df.dropna(subset=[target])
    df['Target'] = df[target].apply(lambda x: 1 if str(x).strip().upper() == 'BREAKDOWN' else 0)

    if len(df['Target'].unique()) < 2:
        print("Warning: Not enough variance in target variable (need both 0 and 1). Model may not be predictive.")
        # Proceed anyway so app.py finds a model, but results will be deterministic
    
    X = df[features]
    y = df['Target']

    # Safe split
    if len(df) < 10:
        print("Dataset very small. Using all data for training.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        # Avoid train_test_split error if one class is too small
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(df['Target'].unique()) > 1 else None)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Model training failed: {e}")
        return

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        try:
            print(classification_report(y_test, y_pred))
        except ValueError:
            print("Not enough classes for full classification report.")

    # Save model safely
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        joblib.dump(model, model_path)
        print(f"\nModel successfully trained and saved to {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

if __name__ == "__main__":
    train_maintenance_model()
