# ============================================================
# Farzad Zandi, 2026
# One-Hot Encoding of Gene Ontology (GO) Terms
# ============================================================

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading GO annotation data...")
data = pd.read_csv('/human_GO.csv')
# data = pd.read_csv('/virus_GO.csv')

# ------------------------------------------------------------
# Prepare GO annotations
# ------------------------------------------------------------
data = data.drop(data.columns[[0, 1]], axis=1)

data['GO_Terms'] = (
    data['GO_Terms']
    .fillna('')
    .apply(
        lambda x: [
            term for term in x.split('; ')
            if term.strip()
        ] if isinstance(x, str) else []
    )
)

# ------------------------------------------------------------
# One-hot encode GO terms
# ------------------------------------------------------------
mlb = MultiLabelBinarizer()

go_matrix = pd.DataFrame(
    mlb.fit_transform(data['GO_Terms']),
    columns=mlb.classes_
)

go_matrix['Protein_ID'] = data['Protein_ID']

# ------------------------------------------------------------
# Remove proteins without GO annotations
# ------------------------------------------------------------
go_matrix['GO_NA'] = (go_matrix[mlb.classes_].sum(axis=1) == 0)
print("Proteins without GO annotations:", go_matrix['GO_NA'].sum())

go_matrix = go_matrix[
    ~go_matrix['GO_NA']
].drop(columns=['GO_NA'])

# ------------------------------------------------------------
# Save encoded GO representations
# The same procedure can be applied to viral proteins.
# ------------------------------------------------------------
go_matrix.to_csv('/human_go_encoded.csv', index=False)
# go_matrix.to_csv('/virus_go_encoded.csv', index=False)

print("GO one-hot encoding completed.")
