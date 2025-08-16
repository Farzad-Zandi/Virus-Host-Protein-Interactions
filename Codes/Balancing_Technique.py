## Farzad Zandi, 2025.
# Balancing Techniques.
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

print("=====================================")
print("Farzad Zandi, 2025.")
print("Balancing Dataset.")

print("Loading Data...")
dataset = pd.read_csv('/data1_CT.csv')
N = dataset.shape[1]-1
target = dataset.iloc[:,N]
dataset = dataset.drop(dataset.columns[[0,N]], axis=1)

print("Data Dimension: ", dataset.shape)
target = pd.DataFrame(target)
print("Normalizing Data...")
dataset = preprocessing.minmax_scale(dataset, feature_range=(0,1))
dataset = pd.DataFrame(dataset)

X = dataset
y = target

# Balancing dataset.
print("Balancing data by SMOTE...")
imbModel = SMOTE(random_state=42)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by ADASYN...")
imbModel = ADASYN(random_state=130)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by Smote Tomek...")
imbModel = SMOTETomek(random_state=139)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by Smote ENN...")
imbModel = SMOTEENN(random_state=73)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by Border Line Smote...")
imbModel = BorderlineSMOTE(random_state=77)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by Random Over Sampler...")
imbModel = RandomOverSampler(random_state=22)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by Random Under Sampler...")
imbModel = RandomUnderSampler(random_state=97)
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)

print("Balancing data by NearMiss...")
imbModel = NearMiss()
X_imb, y_imb = imbModel.fit_resample(X, y)
dataset = X_imb
target = y_imb
print("Transfered dimension: ", dataset.shape)