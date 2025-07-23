
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from models.mlp_model import MLPClassifier

# Charge the DATA
df = pd.read_csv("data/features_dataset.csv")
df["label"] = df["label"].apply(lambda x: 0 if x.strip().lower() == "benign" else 1)
urls = df["url"].values

X = df.drop(columns=["label", "url"]).values
y = df["label"].values.reshape(-1, 1)
X_raw = X.copy()

# Normalisation for MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Split train/test
X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
    X_scaled, y, urls, test_size=0.2, random_state=42
)
X_train_raw, X_test_raw, _, _, _, _ = train_test_split(
    X_raw, y, urls, test_size=0.2, random_state=42
)

# MLP training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

input_dim = X.shape[1]
model = MLPClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training MLP model...")
for epoch in range(10):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:, 1].unsqueeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "mlp_model.pt")
print("MLP model saved as mlp_model.pt")

# Evaluation of MLP
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    preds = model(X_test_tensor)[:, 1].numpy()
    preds_bin = (preds >= 0.5).astype(int)

mlp_acc = accuracy_score(y_test, preds_bin)
mlp_cm = confusion_matrix(y_test, preds_bin)
print("\n MLP Evaluation:")
print(f"Accuracy: {mlp_acc:.4f}")
print("Confusion Matrix:\n", mlp_cm)

# Random Forest training
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_raw, y_train.ravel())
joblib.dump(rf_model, "rf_model.pkl")
print("Random Forest model saved as rf_model.pkl")

# Evaluation of Random Forest
rf_preds = rf_model.predict(X_test_raw)
rf_acc = accuracy_score(y_test, rf_preds)
rf_cm = confusion_matrix(y_test, rf_preds)
print("\nRandom Forest Evaluation:")
print(f"Accuracy: {rf_acc:.4f}")
print("Confusion Matrix:\n", rf_cm)

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(mlp_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("MLP - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()
print("📊 Confusion matrices saved as confusion_matrices.png")

# 100 random URLs
sampled_urls = random.sample(list(urls_test), min(100, len(urls_test)))
with open("test_urls.txt", "w", encoding="utf-8") as f:
    for url in sampled_urls:
        f.write(url + "\n")
print("📁 100 Random URLs for testing saved to test_urls.txt")
