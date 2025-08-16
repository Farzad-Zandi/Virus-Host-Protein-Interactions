## Farzad Zandi, 2025.
# One-Hot Encoding.

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading data...")
data = pd.read_csv('/human_GO.csv')
# data = pd.read_csv('/virus_GO.csv')


# Drop unnecessary index column if present
data = data.drop(data.columns[[0,1]], axis=1)

# Clean and split GO terms
data['GO_Terms'] = data['GO_Terms'].fillna('').apply(
    lambda x: [term for term in x.split('; ') if term.strip()] if isinstance(x, str) else [])

# One-hot encoding
mlb = MultiLabelBinarizer()
go_matrix = pd.DataFrame(mlb.fit_transform(data['GO_Terms']), columns=mlb.classes_)
go_matrix['Protein_ID'] = data['Protein_ID']

# Identify proteins with no GO terms
go_matrix['GO_NA'] = (go_matrix[mlb.classes_].sum(axis=1) == 0)
print("Samples with no GO terms:", go_matrix['GO_NA'].sum())

# Remove proteins with no GO terms
go_matrix = go_matrix[~go_matrix['GO_NA']].drop(columns=['GO_NA'])

# go_matrix.to_csv('/virus_go_encoded.csv')
go_matrix.to_csv('/human_go_encoded.csv')