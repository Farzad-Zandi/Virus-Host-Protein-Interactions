## Farzad Zandi, 2025.
# Feature Selection Methods.

import os
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn import manifold
from sklearn.manifold import MDS
from pydiffmap import diffusion_map as dm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import NMF
from factor_analyzer import FactorAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score
import cma
import optuna
from optuna.samplers import TPESampler
import logging

warnings.filterwarnings('ignore')
print("Farzad Zandi, 2025.")
print("Feature Selection...")
print("Loading data...")
data = pd.read_csv('/data1_CT.csv')
N = data.shape[1]-1
label = data.iloc[:, N-1]
data = data.drop(data.columns[[0, N-1, N]], axis=1)
label = np.where(label==-1, 0, label)
print("Data Dimension: ", data.shape)
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))

#########################################################################
""" NCA feature selection"""
print("Feature selection by Neighborhood Components Analysis...")
model = NeighborhoodComponentsAnalysis(n_components=int(data.shape[1]/2), random_state=42, max_iter=1)
dataNCA = model.fit_transform(data, label)
#########################################################################
""" LLE feature selection"""
print("Feature selection by Locally Linear Embedding...")
model = LocallyLinearEmbedding(n_components=int(data.shape[1]/2))
dataLLE = model.transform(data)
#########################################################################
""" IsoMAP feature selection"""
model = manifold.Isomap(n_components=int(data.shape[1]/2), n_neighbors=3)
dataIso = model.fit_transform(data)
#########################################################################
""" DMAP feature selection"""
model = dm.DiffusionMap.from_sklearn()
dataDMAP = model.fit_transform(data)
#########################################################################
""" MDS feature selection"""
model = MDS(n_components=int(data.shape[1]/2), metric=False)
dataMDS = model.fit_transform(data)
#########################################################################
""" PCA feature selection"""
model = PCA(n_components=int(data.shape[1]/2))
dataPCA = model.fit_transform(data)
#########################################################################
""" Kernel PCA feature selection"""
model = KernelPCA(n_components=int(data.shape[1]/2))
dataKPCA = model.fit_transform(data)
#########################################################################
""" NMF feature selection"""
model = NMF(n_components=int(data.shape[1]/2), init='random', random_state=0)
dataNMF = model.fit_transform(data)
#########################################################################
""" FA feature selection"""
model = FactorAnalyzer(n_factors=int(data.shape[1]/2), rotation="varimax") # rotation = promax
dataFA = model.fit_transform(data)
#########################################################################
""" Monte Carlo feature selection"""
def mcfs_feature_selection(X, y, num_subsets=100, subset_size=5):
    num_features = X.shape[1]
    feature_scores = np.zeros(num_features)
    for _ in range(num_subsets):
        selected_features = np.random.choice(num_features, size=subset_size, replace=False)
        X_subset = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
        model =  YOUR_MODEL
        model.fit(X_train, y_train)
        feature_importance = model.feature_importances_
        feature_scores[selected_features] += feature_importance
    feature_scores /= num_subsets
    ranked_features = np.argsort(feature_scores)[::-1]
    return ranked_features

ranked_features = mcfs_feature_selection(data, label)
dataMonteCarlo = data[:,ranked_features[:int(len(ranked_features)/2)]]
#########################################################################
""" Optuna"""
class FeatureSelectionOptuna:
    def __init__(self, features, penalty=0):
        self.features = features
        self.penalty = penalty

    def __call__(self, trial: optuna.trial.Trial):
        selected_features = [trial.suggest_categorical(name, [True, False]) for name in self.features]
        selected_feature_names = [name for name, selected in zip(self.features, selected_features) if selected]
        n_used = len(selected_feature_names)
        total_penalty = n_used * self.penalty
        loss = 0
        data_selected = data[:,selected_feature_names].copy()
        acc = np.mean(cross_val_score(model, data_selected, np.array(label).ravel(), scoring='accuracy', cv=5))
        loss = -acc
        loss += total_penalty
        return loss

optuna.logging.set_verbosity(logging.WARNING)
features = list(pd.DataFrame(data).columns)
sampler = TPESampler(seed = 32)
study = optuna.create_study(direction="minimize",sampler=sampler)
default_features = {ft: True for ft in features}
study.enqueue_trial(default_features)
study.optimize(FeatureSelectionOptuna(features=features, penalty = 1e-4), n_trials=100)
selected_features = study.best_params
selected_features = true_indices = [index for index, value in selected_features.items() if value]
dataOptuna = data[:, selected_features]

