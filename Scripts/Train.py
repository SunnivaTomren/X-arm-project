import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# ── Load data ─────────────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(_here, "..", "data", "processed", "features.xlsx"))

FEATURES = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
            "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]

X = df[FEATURES].values
y = LabelEncoder().fit_transform(df["label"])   # rest/opening/closing → 0/1/2

# ── Split & normalise ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model = nn.Sequential(
    nn.Linear(13, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 3)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0., 1.0], device=device))  # equal weights for all classes

# ── Train ─────────────────────────────────────────────────────────────────────

for epoch in range(1, 100):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
    print(f"Epoch {epoch:02d}/30  loss: {total_loss/len(y_train):.4f}  acc: {correct/len(y_train)*100:.1f}%")

# ── Evaluate ──────────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()

print("\n", classification_report(y_test, preds, target_names=["closing", "opening", "rest"]))

# ── Save ──────────────────────────────────────────────────────────────────────

import os
save_dir = os.path.join(os.path.dirname(__file__), "..", "Models")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "emg_model.pt")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {os.path.abspath(save_path)}")