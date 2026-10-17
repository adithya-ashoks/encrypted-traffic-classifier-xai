
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Project folders
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

print("Loading datasets...")
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = np.load(DATA / "y_train.npy")
y_test = np.load(DATA / "y_test.npy")

with open(DATA / "label_mapping.json", encoding="utf-8") as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)
expected_classes = np.arange(num_classes)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Number of classes:", num_classes)

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and test feature counts differ.")

if not np.array_equal(np.unique(y_train), expected_classes):
    raise ValueError("Training labels do not match the label mapping.")

if not np.array_equal(np.unique(y_test), expected_classes):
    raise ValueError("Test labels do not match the label mapping.")

# Load saved models
print("\nLoading models...")
rf = joblib.load(MODELS / "rf_model.pkl")
xgb = joblib.load(MODELS / "xgboost_model.pkl")
cnn = load_model(MODELS / "cnn_model.h5", compile=False)
lstm = load_model(MODELS / "lstm_model.h5", compile=False)

# Verify the class order for the tree models
for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError(f"{name} class order does not match label mapping.")

# Fit scalers ONLY on training data, then transform test data
print("\nPreparing CNN and LSTM inputs...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cnn_input = X_test_scaled.reshape(
    X_test_scaled.shape[0], X_test_scaled.shape[1], 1
)
lstm_input = X_test_scaled.reshape(
    X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
)

# Generate probabilities
probabilities = {}

print("\nPredicting with Random Forest...")
probabilities["Random Forest"] = rf.predict_proba(X_test)

print("Predicting with XGBoost...")
probabilities["XGBoost"] = xgb.predict_proba(X_test)

print("Predicting with CNN...")
probabilities["CNN"] = cnn.predict(cnn_input, verbose=0)

print("Predicting with LSTM...")
probabilities["LSTM"] = lstm.predict(lstm_input, verbose=0)

# Check probability dimensions and values
for name, probs in probabilities.items():
    if probs.shape != (len(y_test), num_classes):
        raise ValueError(
            f"{name} returned unexpected probability shape: {probs.shape}"
        )
    if not np.isfinite(probs).all():
        raise ValueError(f"{name} returned non-finite probabilities.")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(f"{name} probabilities do not sum to 1.")

# Equal-weight soft voting
print("\nCalculating equal-weight soft voting...")
ensemble_probs = np.mean(
    np.stack(list(probabilities.values()), axis=0), axis=0
)

probabilities["Soft Voting Ensemble"] = ensemble_probs

# Evaluate all models on the same held-out test set
rows = []
for name, probs in probabilities.items():
    predictions = np.argmax(probs, axis=1)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    })

results = pd.DataFrame(rows).sort_values(
    "Macro F1", ascending=False
)

output_file = RESULTS / "soft_voting_results.csv"
results.to_csv(output_file, index=False)

print("\n=== MODEL COMPARISON ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nResults saved to:", output_file)
print("Evaluation complete.")
