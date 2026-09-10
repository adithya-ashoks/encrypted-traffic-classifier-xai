# Encrypted Traffic Classifier XAI

ML-based classification of encrypted network traffic using flow-level statistical features and Explainable AI (XAI).

## Overview

Encrypted network traffic limits the usefulness of traditional payload-based inspection techniques. This project investigates the classification of encrypted network traffic using statistical features extracted at the network-flow level, without relying on direct payload inspection.

The project evaluates multiple machine learning and deep learning approaches and incorporates Explainable AI techniques to improve the interpretability of classification decisions.

A key objective of the project is to investigate not only classification performance within individual datasets, but also the ability of trained models to generalize across different traffic datasets.

## Objectives

* Classify encrypted network traffic using flow-level statistical features.
* Compare the performance of different machine learning and deep learning models.
* Evaluate model performance using standard classification metrics.
* Investigate class-level performance and misclassification patterns.
* Apply Explainable AI techniques to understand model predictions and important features.
* Evaluate model generalizability across different network traffic datasets.

## Datasets

### ISCX VPN-nonVPN

The primary dataset used during the initial development phase is the ISCX VPN-nonVPN flow-level traffic dataset.

The processed dataset used in the project contains:

* 59,706 flow records
* 23 flow-level features
* 14 traffic categories

The traffic categories include:

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

The dataset was verified for missing values before preprocessing.

### CIC-Darknet2020

CIC-Darknet2020 is used as a second dataset for evaluating model performance on an independent traffic source and for investigating cross-dataset generalizability.

Feature compatibility between the datasets is evaluated before conducting cross-dataset experiments.

## Data Preprocessing

The preprocessing pipeline consists of the following steps:

1. Load and validate the raw dataset.
2. Verify the dataset for missing values and structural inconsistencies.
3. Separate input features from the target traffic label.
4. Encode the categorical traffic labels numerically.
5. Perform a stratified 80:20 train-test split.
6. Use `random_state=42` to ensure reproducibility.
7. Save the processed training and testing data for consistent use across the project.

The processed ISCX dataset is stored as:

```text
X_train.csv
X_test.csv
y_train.npy
y_test.npy
label_mapping.json
```

Using a shared processed dataset ensures that all models are evaluated using the same train-test partition.

## Models

The project evaluates four model architectures:

### Random Forest

Random Forest is used as the baseline tree-based machine learning model. It provides a reference point for comparing the performance of more complex models.

### XGBoost

XGBoost is evaluated as a gradient-boosting based model for encrypted traffic classification.

### 1D Convolutional Neural Network (CNN)

A 1D-CNN is evaluated as a deep learning approach for learning patterns from flow-level feature representations.

The current CNN architecture consists of:

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

StandardScaler is applied before CNN training, and EarlyStopping is used during training to prevent unnecessary training after validation performance stops improving.

### Long Short-Term Memory (LSTM)

An LSTM-based model is included to investigate whether a sequential deep learning architecture provides advantages when applied to flow-level traffic features.

## Explainable AI

Explainable AI is incorporated to investigate how the models arrive at their classification decisions.

The project uses SHAP-based techniques to analyze feature importance and individual predictions.

Tree-based models use TreeSHAP, while deep learning models use appropriate SHAP-based approaches.

The explainability analysis focuses on both:

* **Global explanations** — identifying features that have the greatest influence on model predictions across the dataset.
* **Local explanations** — examining the features influencing the prediction of individual traffic flows.

Particular attention is given to classes that exhibit higher levels of confusion during classification.

## Evaluation

Model performance is evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Per-class performance is also examined to identify traffic categories that are more difficult to distinguish.

## Experimental Design

The project consists of two major evaluation settings.

### In-Dataset Evaluation

Each model is trained and evaluated within the same dataset to establish its classification performance under standard conditions.

### Cross-Dataset Evaluation

Models trained on the ISCX dataset are evaluated directly on CIC-Darknet2020 without retraining.

This experiment investigates whether the learned traffic representations generalize to an independent dataset and helps identify potential dependence on the characteristics of a single training dataset.

## Reproducibility

The project uses a controlled Python environment and a fixed random seed for the primary train-test split.

The preprocessing stage is performed once and the resulting datasets are stored for shared use across the team.

The repository also maintains the notebooks, processed data references, model implementations, and experiment results required to reproduce the study.

## Project Structure

```text
encrypted-traffic-classifier-xai/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── processed/
│
├── notebooks/
│   ├── preprocessing/
│   ├── model_training/
│   └── evaluation/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── confusion_matrices/
│
└── docs/
```

The directory structure may be updated as additional experiments and datasets are integrated.

## Results

Experimental results for the individual models, dataset comparisons, and cross-dataset evaluation will be consolidated in this section as the experiments are completed.

The final comparison will include performance across all four models and both datasets, together with cross-dataset generalization results.

## Installation

The project uses Python 3.10 and was developed in a Conda environment.

Create and activate the environment using:

```bash
conda create -n traffic-classifier python=3.10
conda activate traffic-classifier
```

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up the required Python environment.
2. Place the required datasets in the appropriate data directory.
3. Run the preprocessing notebook to generate the processed datasets.
4. Run the individual model notebooks to train and evaluate the models.
5. Run the explainability notebooks after model training.
6. Store evaluation results and visualizations in the designated results directories.

## Current Status

The project currently includes:

* Environment and repository setup
* Dataset validation and preprocessing
* Label encoding for 14 traffic categories
* Stratified 80:20 train-test split
* Shared processed training and testing datasets
* Random Forest baseline
* 1D-CNN implementation
* Initial model evaluation
* Explainability and second-dataset evaluation planned as subsequent stages

## Contributors

This project is developed as a collaborative research project involving work in:

* Data preprocessing and reproducibility
* Machine learning model development
* Deep learning model development
* Explainable AI
* Cross-dataset evaluation
* Documentation and research methodology

## License

This project is intended for academic and research purposes.
