# ============================================================
# Farzad Zandi, 2026
# Feature Selection and Dimensionality Reduction Methods
# ============================================================

import warnings
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA, NMF
from factor_analyzer import FactorAnalyzer
from sklearn.manifold import SpectralEmbedding
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings('ignore')
print("=" * 60)
print("Farzad Zandi, 2026")
print("Feature Selection and Dimensionality Reduction Methods")
print("=" * 60)

# ============================================================
# Load and Preprocess the Dataset
# ============================================================

print("\nLoading dataset...")
data = pd.read_csv('/DATASET_NAME.csv')

# Identify the target and auxiliary columns
N = data.shape[1] - 1
label = data.iloc[:, N - 1]

# Remove identifier, label, and auxiliary columns
data = data.drop(data.columns[[ ]], axis=1)

# Convert negative class labels to zero
label = np.where(label == -1, 0, label)
print(f"Dataset dimensions: {data.shape}")

# ------------------------------------------------------------
# Normalize feature values
# ------------------------------------------------------------
print("Normalizing feature values...")
data = preprocessing.minmax_scale(data, feature_range=(0, 1))

# ============================================================
# Principal Component Analysis (PCA)
# ============================================================
print("\nApplying Principal Component Analysis (PCA)...")
model = PCA(n_components=int(data.shape[1] / 2))
dataPCA = model.fit_transform(data)

# ============================================================
# Non-Negative Matrix Factorization (NMF)
# ============================================================
print("Applying Non-Negative Matrix Factorization (NMF)...")
model = NMF(
    n_components=int(data.shape[1] / 2),
    init='random',
    random_state=0
)
dataNMF = model.fit_transform(data)

# ============================================================
# Factor Analysis (FA)
# ============================================================
print("Applying Factor Analysis (FA)...")
model = FactorAnalyzer(
    n_factors=int(data.shape[1] / 2),
    rotation="varimax"
)

dataFA = model.fit_transform(data)

# ============================================================
# Multi-Cluster Feature Selection (MCFS)
# ============================================================
print("Applying Multi-Cluster Feature Selection (MCFS)...")
embedding = SpectralEmbedding(
    n_components=5,
    n_neighbors=10,
    affinity='nearest_neighbors',
    random_state=42
).fit_transform(data)

scores = np.zeros(data.shape[1])

for i in range(embedding.shape[1]):
    model = Lasso(alpha=0.01, max_iter=10000)
    model.fit(DATA_NAME, embedding[:, i])
    scores += np.abs(model.coef_)

ranked_features = np.argsort(scores)[::-1]
dataMCFS = data[:, ranked_features[:int(data.shape[1] / 2)]]

print("MCFS completed.")
print("Selected features:", dataMCFS.shape[1])

# ============================================================
# Optuna-Based Feature Selection
# ============================================================
print("Applying Optuna-based feature selection...")

class FeatureSelectionOptuna:
    def __init__(
        self,
        features,
        penalty=0
    ):
        self.features = features
        self.penalty = penalty

    def __call__(
        self,
        trial: optuna.trial.Trial
    ):

        # Determine whether each feature is selected
        selected_features = [
            trial.suggest_categorical(
                name,
                [True, False]
            )
            for name in self.features
        ]

        selected_feature_names = [
            name
            for name, selected
            in zip(
                self.features,
                selected_features
            )
            if selected
        ]

        # Penalize larger feature subsets
        n_used = len(
            selected_feature_names
        )

        total_penalty = (
            n_used * self.penalty
        )

        data_selected = data[
            :,
            selected_feature_names
        ].copy()

        acc = np.mean(
            cross_val_score(
                model,
                data_selected,
                np.array(label).ravel(),
                scoring='accuracy',
                cv=5
            )
        )

        loss = -acc
        loss += total_penalty

        return loss

optuna.logging.set_verbosity(logging.WARNING)
features = list(pd.DataFrame(data).columns)
sampler = TPESampler(seed=32)
study = optuna.create_study(direction="minimize", sampler=sampler)
default_features = {feature: True for feature in features}
study.enqueue_trial(default_features)
study.optimize(
    FeatureSelectionOptuna(
        features=features,
        penalty=1e-4
    ),
    n_trials=100
)
selected_features = study.best_params
selected_features = [
    index
    for index, value
    in selected_features.items()
    if value
]

dataOptuna = data[:, selected_features]

print("\n" + "=" * 60)
print(
    "Feature selection and dimensionality reduction "
    "procedures completed successfully."
)
print("Methods: PCA, NMF, FA, MCFS, and Optuna")
print("=" * 60)
