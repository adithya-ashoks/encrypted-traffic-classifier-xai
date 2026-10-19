
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Scaling features for CNN...")

scaler = StandardScaler()
scaler.fit(X_train.values)
X_test_scaled = scaler.transform(X_test.values)
X_test_cnn = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

print("Loading Random Forest and CNN models...")

rf = joblib.load(MODEL_DIR / "rf_model.pkl")
cnn = load_model(MODEL_DIR / "cnn_model.h5", compile=False)

print("Generating predictions...")

rf_probs = rf.predict_proba(X_test)
cnn_probs = cnn.predict(X_test_cnn, verbose=0)

if rf_probs.shape != cnn_probs.shape:
    raise ValueError(
        f"Probability shape mismatch: RF {rf_probs.shape}, "
        f"CNN {cnn_probs.shape}"
    )

if not np.isfinite(rf_probs).all() or not np.isfinite(cnn_probs).all():
    raise ValueError("Non-finite prediction probabilities detected.")

ensemble_probs = (rf_probs + cnn_probs) / 2

predictions = {
    "Random Forest": np.argmax(rf_probs, axis=1),
    "CNN": np.argmax(cnn_probs, axis=1),
    "RF + CNN Soft Voting": np.argmax(ensemble_probs, axis=1),
}

rows = []

for name, y_pred in predictions.items():
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    })

results = pd.DataFrame(rows)

print("\n=== RANDOM FOREST vs CNN vs TWO-MODEL ENSEMBLE ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

output_path = RESULTS_DIR / "rf_cnn_ensemble_results.csv"
results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
