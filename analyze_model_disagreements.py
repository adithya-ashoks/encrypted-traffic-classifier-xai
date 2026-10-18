
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading test data...")
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_test = np.load(DATA / "y_test.npy")

with open(DATA / "label_mapping.json", encoding="utf-8") as f:
    label_mapping = json.load(f)

print("Loading saved models...")
rf = joblib.load(MODELS / "rf_model.pkl")
xgb = joblib.load(MODELS / "xgboost_model.pkl")
cnn = load_model(MODELS / "cnn_model.h5", compile=False)
lstm = load_model(MODELS / "lstm_model.h5", compile=False)

print("Preparing neural network inputs...")
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_test)

cnn_input = X_scaled.reshape(
    X_scaled.shape[0], X_scaled.shape[1], 1
)
lstm_input = X_scaled.reshape(
    X_scaled.shape[0], 1, X_scaled.shape[1]
)

print("Generating predictions...")
predictions = {
    "Random Forest": rf.predict(X_test),
    "XGBoost": xgb.predict(X_test),
    "CNN": np.argmax(cnn.predict(cnn_input, verbose=0), axis=1),
    "LSTM": np.argmax(lstm.predict(lstm_input, verbose=0), axis=1),
}

# Identify test examples where Random Forest is wrong.
rf_predictions = predictions["Random Forest"]
rf_wrong = rf_predictions != y_test

# Compare individual model accuracy and agreement with Random Forest.
rows = []
for name, pred in predictions.items():
    correct_when_rf_wrong = (
        np.mean(pred[rf_wrong] == y_test[rf_wrong])
        if np.any(rf_wrong)
        else np.nan
    )

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Agreement_with_RF": np.mean(pred == rf_predictions),
        "Correct_when_RF_wrong": correct_when_rf_wrong,
    })

summary = pd.DataFrame(rows)
summary_path = RESULTS / "model_disagreement_summary.csv"
summary.to_csv(summary_path, index=False)

# Generate probabilities for the same equal-weight ensemble.
print("Calculating ensemble predictions...")
rf_probs = rf.predict_proba(X_test)
xgb_probs = xgb.predict_proba(X_test)
cnn_probs = cnn.predict(cnn_input, verbose=0)
lstm_probs = lstm.predict(lstm_input, verbose=0)

ensemble_probs = np.mean(
    np.stack(
        [rf_probs, xgb_probs, cnn_probs, lstm_probs],
        axis=0
    ),
    axis=0
)
ensemble_predictions = np.argmax(ensemble_probs, axis=1)

rf_correct = rf_predictions == y_test
ensemble_correct = ensemble_predictions == y_test

print("\n=== MODEL AGREEMENT SUMMARY ===")
print(summary.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

print("\n=== RANDOM FOREST VS ENSEMBLE ===")
print(
    "RF correct, ensemble wrong:",
    int(np.sum(rf_correct & ~ensemble_correct))
)
print(
    "RF wrong, ensemble correct:",
    int(np.sum(~rf_correct & ensemble_correct))
)
print(
    "Both correct:",
    int(np.sum(rf_correct & ensemble_correct))
)
print(
    "Both wrong:",
    int(np.sum(~rf_correct & ~ensemble_correct))
)

# Count how many individual models predict each example correctly.
correct_matrix = np.column_stack([
    predictions[name] == y_test for name in predictions
])
num_correct = correct_matrix.sum(axis=1)

print("\n=== NUMBER OF MODELS CORRECT PER TEST EXAMPLE ===")
for count in range(5):
    print(
        f"{count} models correct:",
        int(np.sum(num_correct == count))
    )

print("\nSaved:", summary_path)
print("Disagreement analysis complete.")
