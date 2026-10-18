
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

print("Loading test data...")
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_test = np.load(DATA / "y_test.npy")

with open(DATA / "label_mapping.json", encoding="utf-8") as f:
    label_mapping = json.load(f)

# Map numeric class IDs to their names
class_names = [
    name for name, class_id in
    sorted(label_mapping.items(), key=lambda item: item[1])
]
class_ids = list(range(len(class_names)))

print("Loading saved models...")
rf = joblib.load(MODELS / "rf_model.pkl")
xgb = joblib.load(MODELS / "xgboost_model.pkl")
cnn = load_model(MODELS / "cnn_model.h5", compile=False)
lstm = load_model(MODELS / "lstm_model.h5", compile=False)

print("Preparing neural network inputs...")
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_test)

cnn_input = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
lstm_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

print("Generating predictions...")
probabilities = {
    "Random Forest": rf.predict_proba(X_test),
    "XGBoost": xgb.predict_proba(X_test),
    "CNN": cnn.predict(cnn_input, verbose=0),
    "LSTM": lstm.predict(lstm_input, verbose=0),
}

probabilities["Soft Voting Ensemble"] = np.mean(
    np.stack(list(probabilities.values()), axis=0), axis=0
)

# Per-class precision, recall, F1 and support
reports = []
for model_name, probs in probabilities.items():
    predictions = np.argmax(probs, axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    for class_name in class_names:
        metrics = report[class_name]
        reports.append({
            "Model": model_name,
            "Class": class_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1-score"],
            "Support": metrics["support"],
        })

per_class = pd.DataFrame(reports)
output_path = RESULTS / "ensemble_per_class_results.csv"
per_class.to_csv(output_path, index=False)

# Compare ensemble against the strongest individual model per class
wide = per_class.pivot(index="Class", columns="Model", values="F1")
individual_models = ["Random Forest", "XGBoost", "CNN", "LSTM"]
wide["Best Individual F1"] = wide[individual_models].max(axis=1)
wide["Ensemble F1 Change"] = (
    wide["Soft Voting Ensemble"] - wide["Best Individual F1"]
)

comparison_path = RESULTS / "ensemble_class_comparison.csv"
wide.reset_index().to_csv(comparison_path, index=False)

print("\n=== ENSEMBLE VS BEST INDIVIDUAL MODEL BY CLASS ===")
print(
    wide[["Best Individual F1", "Soft Voting Ensemble",
          "Ensemble F1 Change"]]
    .sort_values("Ensemble F1 Change")
    .to_string(float_format=lambda value: f"{value:.4f}")
)

improved = (wide["Ensemble F1 Change"] > 0.0001).sum()
worsened = (wide["Ensemble F1 Change"] < -0.0001).sum()
unchanged = len(wide) - improved - worsened

print(f"\nClasses improved: {improved}")
print(f"Classes worsened: {worsened}")
print(f"Classes effectively unchanged: {unchanged}")
print("\nSaved:")
print(output_path)
print(comparison_path)
print("\nAnalysis complete.")
