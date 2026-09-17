# Encrypted Traffic Classifier XAI

**ML-based classification of encrypted network traffic using flow-level statistical features and Explainable AI (XAI).**

## Overview

Encrypted network traffic limits the effectiveness of traditional payload-based inspection techniques because the contents of the communication are not directly available for inspection.

This project investigates the classification of encrypted network traffic using **flow-level statistical features**, without relying on direct payload inspection.

The study systematically compares **traditional machine learning models** with **deep learning architectures** using the same statistical feature representation:

* Random Forest
* XGBoost
* 1D-CNN
* LSTM

In addition to in-dataset classification, the project investigates **model explainability** and **cross-dataset generalization** using CIC-Darknet2020.

A central research question is whether conventional tree-based models are better suited to the available flow-level statistical representation than the evaluated neural architectures.

---

## Objectives

The main objectives of the project are to:

* Classify encrypted network traffic using flow-level statistical features.
* Compare conventional machine learning and deep learning approaches.
* Evaluate model performance using standard classification metrics.
* Analyze class-level performance and misclassification patterns.
* Investigate the effect of class imbalance on model performance.
* Apply Explainable AI techniques to understand model predictions and important features.
* Evaluate model generalization across different encrypted-traffic datasets.
* Investigate how model architecture interacts with flow-level statistical representations.

---

# Datasets

## 1. ISCX VPN-nonVPN

The primary dataset used for model development and in-dataset evaluation is the **ISCX VPN-nonVPN flow-level traffic dataset**.

The processed dataset contains:

* **59,706 flow records**
* **23 flow-level statistical features**
* **14 traffic categories**
* No missing values in the processed dataset

### Traffic Categories

* BROWSING
* VPN-BROWSING
* CHAT
* VPN-CHAT
* VOIP
* VPN-VOIP
* P2P
* VPN-P2P
* FT
* VPN-FT
* MAIL
* VPN-MAIL
* STREAMING
* VPN-STREAMING

### Train-Test Split

The dataset is divided using a stratified **80:20 train-test split** with:

```text
random_state = 42
```

This ensures that all four models use the same training and testing partition.

---

## 2. CIC-Darknet2020

**CIC-Darknet2020** is used as a second dataset to investigate **cross-dataset generalization**.

The dataset contains network traffic associated with different application categories and network types, including Tor, VPN, Non-Tor, and NonVPN traffic.

Because the label taxonomy of CIC-Darknet2020 differs from the 14-class ISCX taxonomy, direct 14-class comparison is not performed.

Instead, a common application-level taxonomy is used:

* Browsing
* Chat
* Email
* File Transfer
* P2P
* Streaming
* VoIP

### Cross-Dataset Evaluation

Models trained on ISCX are evaluated directly on CIC-Darknet2020 **without retraining**.

After cleaning invalid and non-finite records, **158,566 valid samples** were used for the cross-dataset experiment.

The purpose of this experiment is to measure how well models trained on one traffic dataset transfer to another dataset.

---

# Data Preprocessing

The preprocessing pipeline consists of:

1. Loading and validating the raw dataset.
2. Checking for missing, infinite, and structurally invalid values.
3. Separating input features from the target label.
4. Encoding categorical traffic labels using `LabelEncoder`.
5. Performing a stratified 80:20 train-test split.
6. Using `random_state=42` for reproducibility.
7. Saving the processed data for consistent use across models.

The processed ISCX dataset contains:

```text
data/
└── processed/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.npy
    ├── y_test.npy
    └── label_mapping.json
```

All four models use the same processed ISCX train-test split.

---

# Feature Representation

The project uses **23 flow-level statistical features**.

| Feature Group            | Examples                         |
| ------------------------ | -------------------------------- |
| **Flow characteristics** | Duration, packets/sec, bytes/sec |
| **Forward traffic**      | Total, min/max/mean Forward IAT  |
| **Backward traffic**     | Total, min/max/mean Backward IAT |
| **Flow timing**          | Min/max/mean/std Flow IAT        |
| **Active periods**       | Min/mean/max/std Active          |
| **Idle periods**         | Min/mean/max/std Idle            |

For cross-dataset evaluation, the corresponding CIC-Darknet2020 features are mapped to the same 23-feature representation used by the ISCX models.

---

# Models

The project evaluates four model architectures.

## Random Forest

Random Forest is used as the primary conventional machine learning baseline.

Configuration:

```text
n_estimators = 100
random_state = 42
```

Random Forest does not require feature scaling for the baseline experiment.

### Class-Weighted Experiment

A class-weighted Random Forest was also evaluated to investigate the effect of class imbalance.

Results:

```text
Accuracy  ≈ 90.88%
Macro F1  ≈ 0.8959
```

Class weighting did not produce a significant improvement over the original Random Forest baseline.

---

## XGBoost

XGBoost is evaluated as a second conventional tree-based machine learning model.

Configuration:

```text
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

The model is trained directly on the flow-level statistical features without feature scaling.

---

## 1D Convolutional Neural Network

A 1D-CNN is evaluated as a deep learning approach using the same 23-feature representation.

The current architecture is:

```text
Conv1D(64)
    ↓
MaxPooling1D
    ↓
Conv1D(128)
    ↓
MaxPooling1D
    ↓
Flatten
    ↓
Dense(64)
    ↓
Dropout(0.3)
    ↓
Dense(14, Softmax)
```

StandardScaler is applied before CNN training.

EarlyStopping is used during training to prevent unnecessary training once validation performance stops improving.

---

## Long Short-Term Memory

An LSTM-based architecture is included to investigate a recurrent neural-network approach to the flow-level feature representation.

The current architecture is:

```text
LSTM(64)
    ↓
Dropout(0.3)
    ↓
Dense(64, ReLU)
    ↓
Dense(14, Softmax)
```

The 23 features are represented as a single timestep with 23 features.

StandardScaler is applied before LSTM training.

---

# Explainable AI

Explainable AI is incorporated to investigate how different models arrive at their classification decisions.

### Tree-Based Models

**TreeSHAP** is used for Random Forest and XGBoost to investigate:

* Global feature importance
* Feature contribution to predictions
* Class-specific feature behaviour
* Individual prediction explanations

### Deep Learning Models

SHAP-based approaches are being investigated for the CNN and LSTM models to analyze feature contributions to neural-network predictions.

### Explainability Focus

Particular attention is given to:

* Highly influential flow-level features
* Difficult traffic categories
* Commonly confused classes
* Differences between tree-based and neural models

The XAI analysis is intended to help explain **why different model architectures behave differently on the same statistical representation**.

---

# Evaluation

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Weighted F1-score
* Confusion Matrix

Per-class performance is also analyzed to identify traffic categories that are more difficult to distinguish.

---

# Experimental Design

The project consists of two major evaluation settings.

## 1. In-Dataset Evaluation

Models are trained and evaluated using the ISCX VPN-nonVPN dataset.

The same train-test partition and feature representation are used across the four models to provide a consistent comparison.

The purpose is to compare the behaviour of:

**Traditional ML**

Random Forest → XGBoost

**Deep Learning**

1D-CNN → LSTM

The objective is not simply to maximize the performance of every architecture, but to investigate how different model families perform on the same flow-level statistical representation.

---

## 2. Cross-Dataset Evaluation

Models trained on ISCX are evaluated directly on CIC-Darknet2020 **without retraining**.

Because the two datasets use different label taxonomies, predictions and ground-truth labels are mapped to a common application-level taxonomy before evaluation.

### Cross-Dataset Results

| Model             | Application-Level Accuracy |
| ----------------- | -------------------------: |
| **Random Forest** |                 **29.26%** |
| **XGBoost**       |                 **22.92%** |
| **1D-CNN**        |                 **13.55%** |
| **LSTM**          |                 **10.94%** |

These results indicate substantial performance degradation when models are transferred between datasets without retraining.

The tree-based models retain higher application-level accuracy than the evaluated neural architectures in this cross-dataset experiment.

Importantly, these values represent **cross-dataset transfer performance**, not the normal held-out ISCX test accuracy.

---

# Cross-Model Agreement

Model agreement was also examined during the CIC-Darknet2020 cross-dataset experiment.

### Pairwise Agreement

| Model Pair               | Agreement |
| ------------------------ | --------: |
| Random Forest vs XGBoost |    65.99% |
| Random Forest vs CNN     |    30.38% |
| Random Forest vs LSTM    |    26.41% |
| XGBoost vs CNN           |    41.55% |
| XGBoost vs LSTM          |    34.05% |
| CNN vs LSTM              |    85.51% |

All four models produced the same application-level prediction for **20.77%** of the evaluated samples.

The high CNN-LSTM agreement despite their low cross-dataset accuracy demonstrates that **prediction agreement does not necessarily imply prediction correctness**.

---

# Results

## ISCX Baseline

The Random Forest baseline achieved:

```text
Accuracy       : 90.9%
Macro F1-score : 0.90
Weighted F1    : 0.91
```

The 1D-CNN achieved approximately **63% accuracy** on the same ISCX test setting.

The remaining consolidated ISCX metrics for XGBoost and LSTM are being incorporated into the final model comparison.

---

## Research Findings So Far

The experiments currently indicate:

1. Tree-based models provide strong performance on the 23-feature flow-level statistical representation.
2. The evaluated neural architectures show lower in-dataset performance than the Random Forest baseline.
3. Class weighting alone does not substantially resolve difficult-class confusion.
4. All models experience substantial performance degradation during zero-retraining cross-dataset transfer.
5. Tree-based models retain higher application-level accuracy than the evaluated neural models during the Darknet transfer experiment.
6. The observed results motivate further investigation using XAI and error analysis.

These findings are interpreted specifically within the **datasets, feature representation, and model architectures evaluated in this study**.

---

# Reproducibility

The project uses a controlled Python environment and fixed random seeds for reproducible experiments.

The main ISCX train-test split uses:

```text
test_size = 0.20
random_state = 42
stratify = labels
```

The processed training and testing datasets are shared across the model experiments to ensure consistent evaluation.

Model training, preprocessing, compatibility analysis, and evaluation workflows are maintained through Jupyter notebooks in the repository.

Large generated experiment outputs, including the full CIC-Darknet2020 prediction CSV, are excluded from version control where appropriate.

---

# Project Structure

```text
encrypted-traffic-classifier-xai/
│
├── README.md
│
├── data/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.npy
│       ├── y_test.npy
│       └── label_mapping.json
│
├── models/
│   ├── cnn_model.h5
│   └── lstm_model.h5
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_random_forest.ipynb
│   ├── XGBoost_SHAP_Gowri.ipynb
│   └── 03_cic_darknet2020_compatibility_analysis.ipynb
│
├── results/
│   └── ...
│
└── docs/
    └── ...
```

The structure may be expanded as additional experiments and documentation are integrated.

---

# Installation

The project was developed using **Python 3.10** in a Conda environment.

Create the environment:

```bash
conda create -n traffic-classifier python=3.10
```

Activate it:

```bash
conda activate traffic-classifier
```

The main libraries used include:

```text
Python 3.10
NumPy
Pandas
Scikit-learn
TensorFlow
XGBoost
SHAP
Matplotlib
Seaborn
Joblib
```

Additional dependencies may be required depending on the experiment being executed.

---

# Usage

1. Clone the repository.
2. Create and activate the `traffic-classifier` Conda environment.
3. Place the required datasets in the appropriate data directories.
4. Run the preprocessing notebook.
5. Generate the shared processed ISCX dataset.
6. Run the individual model notebooks.
7. Evaluate model performance using the common test set.
8. Run XAI analysis after model training.
9. Run the CIC-Darknet2020 compatibility and cross-dataset evaluation notebook.
10. Store figures and summarized results in the appropriate results directories.

---

# Current Status

### Completed

* Project environment and repository setup
* ISCX dataset validation
* Data preprocessing
* Label encoding
* Stratified 80:20 train-test split
* 23-feature flow-level representation
* Random Forest baseline
* Class-weighted Random Forest experiment
* XGBoost implementation
* 1D-CNN implementation
* LSTM implementation
* Common model evaluation pipeline
* CIC-Darknet2020 integration
* 23-feature cross-dataset compatibility mapping
* Zero-retraining cross-dataset evaluation
* Application-level cross-dataset evaluation
* Cross-model agreement analysis
* GitHub integration of the Darknet evaluation workflow

### In Progress

* Random Forest TreeSHAP analysis
* XGBoost TreeSHAP analysis
* Neural-network explainability
* Detailed class-level error analysis
* Consolidated four-model ISCX comparison
* Final interpretation of ML vs DL behaviour

### Remaining

* Complete XAI analysis
* Consolidate final model comparison
* Analyze important misclassification patterns
* Complete final cross-dataset interpretation
* Prepare final research conclusions and documentation

---

# Research Direction

The project is designed around three connected research questions:

### 1. Model Comparison

How do conventional machine learning models and the evaluated neural architectures perform when using the same flow-level statistical representation?

### 2. Explainability

Which flow-level statistical features influence the classification decisions of different model architectures?

### 3. Generalization

How well do models trained on one encrypted-traffic dataset transfer to another dataset without retraining?

Together, these experiments investigate the relationship between **model architecture, statistical feature representation, explainability, and cross-dataset generalization**.

---

# Contributors

This project is developed collaboratively with contributions spanning:

* Data preprocessing and reproducibility
* Machine learning model development
* Deep learning model development
* Explainable AI
* Cross-dataset evaluation
* Research methodology
* Documentation and analysis

---

# License

This project is intended for academic and research purposes.
