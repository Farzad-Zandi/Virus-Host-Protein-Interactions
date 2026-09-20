# ============================================================
# Farzad Zandi, 2026
# Dataset Balancing Techniques
# ============================================================

import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler)
from imblearn.under_sampling import (RandomUnderSampler, NearMiss)
from imblearn.combine import (SMOTETomek, SMOTEENN)

print("=" * 60)
print("Farzad Zandi, 2026")
print("Dataset Balancing Techniques")
print("=" * 60)

# ------------------------------------------------------------
# Load and preprocess the dataset
# ------------------------------------------------------------
print("\nLoading dataset...")
raw_data = pd.read_csv('/DATA_NAME.csv')

# Preserve the original row index for sample tracking
original_index = np.arange(len(raw_data))

# The last column contains the target labels
N = raw_data.shape[1] - 1
target = raw_data.iloc[:, N]

# Remove identifier and target columns
dataset = raw_data.drop(raw_data.columns[[0, N]], axis=1)
print(f"Dataset dimensions: {dataset.shape}")
target = pd.DataFrame(target)

# ------------------------------------------------------------
# Normalize feature values
# ------------------------------------------------------------
print("\nNormalizing feature values...")
dataset = preprocessing.minmax_scale(dataset, feature_range=(0, 1))
dataset = pd.DataFrame(dataset)

# Add original sample index for tracking
dataset["original_index"] = original_index

# ------------------------------------------------------------
# Separate features and target
# ------------------------------------------------------------
X = dataset.drop(columns=["original_index"])
y = target

# ============================================================
# SMOTE
# ============================================================
print("\nApplying SMOTE...")
imbModel = SMOTE(random_state=42)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# ADASYN
# ============================================================
print("\nApplying ADASYN...")
imbModel = ADASYN(random_state=130)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# SMOTE-Tomek
# ============================================================
print("\nApplying SMOTE-Tomek...")
imbModel = SMOTETomek(random_state=139)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# SMOTE-ENN
# ============================================================
print("\nApplying SMOTE-ENN...")
imbModel = SMOTEENN(random_state=73)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# Borderline-SMOTE
# ============================================================
print("\nApplying Borderline-SMOTE...")
imbModel = BorderlineSMOTE(random_state=77)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# Random Over-Sampling
# ============================================================
print("\nApplying Random Over-Sampling...")
imbModel = RandomOverSampler(random_state=22)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# Random Under-Sampling
# ============================================================
print("\nApplying Random Under-Sampling...")
imbModel = RandomUnderSampler(random_state=97)
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ============================================================
# NearMiss
# ============================================================
print("\nApplying NearMiss...")
imbModel = NearMiss()
X_imb, y_imb = imbModel.fit_resample(X, y)
print(f"Resampled dimensions: {X_imb.shape}")

# ------------------------------------------------------------
# Recover original indices selected by each balancing technique
# ------------------------------------------------------------
selected_indices = imbModel.sample_indices_
print(f"Number of original samples selected by balancing technique: " f"{len(selected_indices)}")

# ------------------------------------------------------------
# Combine balanced features, target labels, and original index
# ------------------------------------------------------------
balancedData = np.c_[X_imb, y_imb, selected_indices]
balancedData = pd.DataFrame(balancedData)

# Rename the last column
balancedData.columns = [*X_imb.columns, "target", "original_index"]

# ------------------------------------------------------------
# Save the NearMiss-balanced dataset
# ------------------------------------------------------------
output_path = ('PATH/BALANCED_DATA.csv')
balancedData.to_csv(output_path, index=False)

print("\nbalanced dataset saved successfully.")
print(f"Output file: {output_path}")

print("=" * 60)
print("Dataset balancing completed successfully.")
print("=" * 60)
