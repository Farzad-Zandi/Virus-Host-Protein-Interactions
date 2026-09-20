# ============================================================
# Farzad Zandi, 2026
# Autoencoder-Based Compression of One-Hot-Encoded
# Gene Ontology (GO) Protein Function Features
# ============================================================

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Farzad Zandi, 2026")
print("Autoencoder-Based Compression of One-Hot-Encoded GO Features")
print("=" * 60)

# ------------------------------------------------------------
# Load and preprocess the dataset
# ------------------------------------------------------------
print("\nLoading dataset...")

data = pd.read_csv('/DATASET_NAME.csv')

# Remove identifier and metadata columns
data = data.drop(data.columns[[]], axis=1)

input_dim = data.shape[1]
encoding_dim = 512
batch_size = 128
epochs = 100

# Select GPU when available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dataset dimensions: {data.shape}")
print(f"Input feature dimension: {input_dim}")
print(f"Encoded feature dimension: {encoding_dim}")
print(f"Computation device: {device}")

# ------------------------------------------------------------
# Prepare data for model training
# ------------------------------------------------------------
data = np.array(data)
tensor_data = torch.from_numpy(data).float()
dataset = TensorDataset(tensor_data)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# ------------------------------------------------------------
# Define the Gene Ontology Autoencoder
# ------------------------------------------------------------
class GOAutoencoder(nn.Module):
    def __init__(self):
        super(GOAutoencoder, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoding_dim)
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, input_dim)
        )

        # Xavier uniform initialization for all linear layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # Encode the input into a lower-dimensional representation
        z = self.encoder(x)

        # Reconstruct the original input from the encoded representation
        out = self.decoder(z)
        return out

# ------------------------------------------------------------
# Configure model training
# ------------------------------------------------------------
model = GOAutoencoder().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

early_stop_counter = 0
patience = 10
loss_hist = []
best_loss = float('inf')
print("\nStarting autoencoder training...")

# ------------------------------------------------------------
# Train the autoencoder
# ------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss_hist.append(train_loss)

    # Early stopping based on training loss
    if train_loss < best_loss:
        best_loss = train_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break
    print(
        f"Epoch {epoch + 1:3d}/{epochs} "
        f"| Training Loss: {train_loss:.4f}"
    )

# ------------------------------------------------------------
# Generate compressed GO representations
# ------------------------------------------------------------
print("\nGenerating compressed GO representations...")
with torch.no_grad():
    encoded_vectors = model.encoder(
        tensor_data.to(device)
    ).cpu()

print("Compression completed successfully.")
print(f"Encoded representation shape: {encoded_vectors.shape}")

# ------------------------------------------------------------
# Save the encoded representations
# ------------------------------------------------------------
encoded_vectors = [
    tensor.detach().numpy()
    for tensor in encoded_vectors
]

encoded_vectors = pd.DataFrame(encoded_vectors)
encoded_vectors.to_csv('/DATASET_NAME_GO_Auto_Encoded.csv', index=False)

print("\nEncoded GO representations saved successfully.")
print("Output file: /DATASET_NAME_GO_Auto_Encoded.csv")
print("=" * 60)
