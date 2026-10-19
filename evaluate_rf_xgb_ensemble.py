
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading datasets...")
X_test = pd.read_csv(DATA / "X_test.csv")
y_test = np.load(DATA / "y_test.npy")

with open(DATA / "label_mapping.json", encoding="utf-8") as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)
expected_classes = np.arange(num_classes)

if X_test.shape[1] != 23:
    raise ValueError(f"Expected 23 features, found {X_test.shape[1]}.")

if not np.array_equal(np.unique(y_test), expected_classes):
    raise ValueError("Test labels do not match the label mapping.")

print("Loading Random Forest and XGBoost...")
rf = joblib.load(MODELS / "rf_model.pkl")
xgb = joblib.load(MODELS / "xgboost_model.pkl")

for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError(f"{name} class order does not match label mapping.")

print("Generating probabilities...")
rf_probs = rf.predict_proba(X_test)
xgb_probs = xgb.predict_proba(X_test)

for name, probs in [
    ("Random Forest", rf_probs),
    ("XGBoost", xgb_probs),
]:
    if probs.shape != (len(y_test), num_classes):
        raise ValueError(f"{name} returned unexpected probability dimensions.")
    if not np.isfinite(probs).all():
        raise ValueError(f"{name} returned non-finite probabilities.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{name} probabilities do not sum to 1.")

print("Calculating equal-weight RF + XGBoost soft voting...")
ensemble_probs = (rf_probs + xgb_probs) / 2

models = {
    "Random Forest": rf_probs,
    "XGBoost": xgb_probs,
    "RF + XGBoost Soft Voting": ensemble_probs,
}

rows = []
for name, probs in models.items():
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

results = pd.DataFrame(rows)
output_file = RESULTS / "rf_xgb_ensemble_results.csv"
results.to_csv(output_file, index=False)

print("\n=== RANDOM FOREST vs XGBOOST vs TWO-MODEL ENSEMBLE ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nResults saved to:", output_file)
