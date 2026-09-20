# ============================================================================
# Farzad Zandi, 2026
# Knowledge Distillation Network for Virus–Host Protein Interaction Prediction
# ============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    auc
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("Farzad Zandi, 20256")
print("Knowledge Distillation Network for Virus–Host Protein Interaction Prediction")
print("=" * 60)

# ============================================================
# Load and Preprocess the Dataset
# ============================================================
print("\nLoading dataset...")
data = pd.read_csv('/DATASET.csv')
label = data.iloc[:, ]

# Remove identifier and metadata columns
data = data.drop(data.columns[[ ]], axis=1)

# Convert negative labels to zero
label = np.where(label == -1, 0, label)
print(f"Data dimensions: {data.shape}")

# ============================================================
# Split Data into Training and Test Sets
# ============================================================
label = np.array(label)
data = np.array(data)
xTrain, xTest, yTrain, yTest = train_test_split(data, label, test_size=0.2, random_state=0)

# Convert data to PyTorch tensors
xTrain = torch.tensor(xTrain, dtype=torch.float32).to(device)
xTest = torch.tensor(xTest, dtype=torch.float32).to(device)
yTrain = torch.tensor(yTrain, dtype=torch.long).to(device)
yTest = torch.tensor(yTest, dtype=torch.long).to(device)

print(f"Training set dimensions: {xTrain.shape}")
print(f"Test set dimensions: {xTest.shape}")

# ============================================================
# Create Data Loader
# ============================================================
train_dataset = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================================================
# Student Model
# ============================================================
class StudentModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stide=2,
            padding=0
        )
        conv_output_size = self._get_conv_output(input_dim)
        
        # Joint branch
       self.fc1_joint = nn.Linear(conv_output_size, hidden_dim)
        selfrelu_joint = nn.LeakyReLU()
        self.fc2_joint = nn.Linear(hidden_dim, output_dim)

        # Marginal branch
        self.fc1_marginal = nn.Linear(conv_output_size, hidden_dim)
        self.relu_marginal = nn.LeakyReLU()
        self.fc2_marginal = nn.Linear(hidden_dim, output_dim
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.rand(1, 1, shape)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        # Joint representation
        joint = self.relu_joint(self.fc1_joint(x))
        joint = self.fc2_joint(joint)

        # Marginal representation
        marginal = self.relu_marginal(self.fc1_marginal(x))
        marginal = self.fc2_marginal(marginal)

        # Conditional representation
        conditional = joint - marginal

        return self.softmax(conditional)

# ============================================================
# Teacher Model: Attention Mechanism
# ============================================================
class TeacherModelAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads=8,
        dropout=0.1
    ):
        super(TeacherModelAttention, self).__init__()
        self.fc1 = nn.Linear( input_dim, hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = x.unsqueeze(1)
        attention_out, _ = self.attention(x, x, x)
        attention_out = attention_out.squeeze(1)
        x = self.layer_norm1(x.squeeze(1) + attention_out)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

# ============================================================
# Teacher Model: Feature Pyramid Network (FPN)
# ============================================================
class TeacherModelFPN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        super(TeacherModelFPN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.final_fc = nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4, output_dim)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        level1 = self.relu(self.fc1(x))
        level2 = self.relu(self.fc2(level1))
        level3 = self.relu(self.fc3(level2))
        aggregated = torch.cat(
            [
                level1,
                level2,
                level3
            ],
            dim=1
        )
        x = self.final_fc(aggregated)
        return self.softmax(x)

# ============================================================
# Teacher Model: Gated Linear Unit (GLU)
# ============================================================
class TeacherModelGLU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        super(TeacherModelGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gate1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.gate2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x) * self.sigmoid(self.gate1(x))
        x = self.fc2(x) * self.sigmoid(self.gate2(x))
        x = self.fc3(x)
        return self.softmax(x)

# ============================================================
# Adaptive Knowledge Distillation Loss
# ============================================================
def adaptive_distillation(student_output, teacher_outputs, target, alpha=0.1):
    criterion = nn.BCELoss()
    distillation_loss = sum(
        nn.KLDivLoss(
            reduction='batchmean'
        )(
            torch.log(student_output),
            teacher_output
        )
        for teacher_output in teacher_outputs
    ) / len(teacher_outputs)

    supervised_loss = criterion(
        student_output,
        F.one_hot(
            target.long(),
            num_classes=2
        ).float()
    )

    return (alpha * distillation_loss + (1 - alpha) * supervised_loss)

# ============================================================
# Train the Student Model
# ============================================================
def train_student(student_model, teacher_models, train_loader, epochs, lr=0.001):
    optimizer = optim.NAdam(student_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10
    )
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 20

    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            student_output = student_model(data)
            teacher_outputs = [teacher(data).detach() for teacher in teacher_models]
            loss = adaptive_distillation(student_output, teacher_outputs, target)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
        print(f"Epoch {epoch + 1}/{epochs} " f"- Loss: {loss.item():.6f}")

        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch " f"{epoch + 1}")
                break
        loss_hist.append(loss.detach().cpu())

# ============================================================
# Initialize the Student and Teacher Networks
# ============================================================
loss_hist = []

input_dim = data.shape[1]
hidden_dim = int(data.shape[1] / 2)
output_dim = 2

student_model = StudentModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
teacher_models = [
    TeacherModelAttention(
        input_dim,
        32,
        output_dim
    ).to(device),
    TeacherModelFPN(
        input_dim,
        hidden_dim,
        output_dim
    ).to(device),
    TeacherModelGLU(
        input_dim,
        hidden_dim,
        output_dim
    ).to(device)
]

# ============================================================
# Model Training
# ============================================================
print("\nTraining Knowledge Distillation network...")
train_student(student_model, teacher_models, train_loader, epochs=100)

# ============================================================
# Save Training Loss
# ============================================================
loss_hist = pd.DataFrame(loss_hist)
loss_hist.to_csv('/DATASET_loss.csv', index=False)
print("Training loss saved successfully.")

# ============================================================
# Model Evaluation
# ============================================================
print("\nEvaluating the student model...")
student_model.eval()
with torch.no_grad():
    yPred_prob = student_model(xTest)
    yPred = torch.argmax(
        yPred_prob,
        dim=1
    ).cpu().numpy()

# ------------------------------------------------------------
# Classification Metrics
# ------------------------------------------------------------
y_test_cpu = yTest.cpu().numpy()
accuracy = accuracy_score(y_test_cpu, yPred)
precision = precision_score(y_test_cpu, yPred)
recall = recall_score(y_test_cpu, yPred)
f1 = f1_score(y_test_cpu, yPred)
mcc = matthews_corrcoef(y_test_cpu, yPred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"MCC: {mcc * 100:.2f}%")

# ------------------------------------------------------------
# Specificity
# ------------------------------------------------------------
cm = confusion_matrix(y_test_cpu, yPred)
tn, fp, fn, tp = cm.ravel()
specificity = (tn / (tn + fp))
print(f"Specificity: {specificity * 100:.2f}%")

# ============================================================
# Precision-Recall Curve and PR-AUC
# ============================================================
yPred_proba = (
    yPred_prob[:, 1]
    .cpu()
    .numpy()
)

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_test_cpu,
        yPred_proba
    )
)

pr_auc = auc(
    recall_curve,
    precision_curve
)

print(f"PR-AUC: {pr_auc * 100:.2f}%")

# ============================================================
# ROC Curve and ROC-AUC
# ============================================================
fpr, tpr, _ = roc_curve(y_test_cpu, yPred_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC: {roc_auc * 100:.2f}%")

# ============================================================
# Save ROC Coordinates
# ============================================================
pd.DataFrame(fpr).to_csv('/DATASET_fpr.csv', index=False)
pd.DataFrame(tpr).to_csv('/DATASET_tpr.csv', index=False)
print("\nROC coordinates saved successfully.")

print("=" * 60)
print("Knowledge Distillation evaluation completed.")
print("=" * 60)
