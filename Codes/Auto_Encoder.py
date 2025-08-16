## Farzad Zandi, 2025.
# Auto Encoding one-hot-encoded protein functions.

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

print("===================")
print("Farzad Zandi, 2025.")
print("Auto Encoding one-hot-encoded protein functions.")

# Load dataset
print("Loading Datasets...")
data = pd.read_csv('/data1_GO.csv')
data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)
input_dim = data.shape[1]
encoding_dim = 512
batch_size = 128
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Data Dimension: ", data.shape)

# Prepare dataset
data = np.array(data)
tensor_data = torch.from_numpy(np.array(data)).float()
dataset = TensorDataset(tensor_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Autoencoder with dropout and Xavier init
class GOAutoencoder(nn.Module):
    def __init__(self):
        super(GOAutoencoder, self).__init__()
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
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, input_dim)
        )

        # Initialize weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Training setup
model = GOAutoencoder().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

early_stop_counter = 0
patience = 10
loss_hist = []
best_loss = float('inf')

print("Auto Encoding...")
# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss_hist.append(train_loss)
    if train_loss < best_loss:
        best_loss = train_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

# Encode final dataset
with torch.no_grad():
    encoded_vectors = model.encoder(tensor_data.to(device)).cpu()

print("Compression complete. Encoded shape:", encoded_vectors.shape)

encoded_vectors = [tensor.detach().numpy() for tensor in encoded_vectors]
encoded_vectors = pd.DataFrame(encoded_vectors)
encoded_vectors.to_csv('d:/data1_GO_Auto_Encoded.csv')