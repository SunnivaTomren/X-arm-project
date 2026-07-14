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
le = LabelEncoder()
y = le.fit_transform(df["label"])
n_classes = len(le.classes_)

# ── Split & normalise ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ── Model (10 lag) ────────────────────────────────────────────────────────────
#
#  Arkitektur: utvid → platå → trekk inn → output
#  BatchNorm stabiliserer trening; Dropout hindrer overfitting.
#
#  Lag  1:  13  → 128   Linear + BN + ReLU + Dropout
#  Lag  2: 128  → 256   Linear + BN + ReLU + Dropout
#  Lag  3: 256  → 256   Linear + BN + ReLU + Dropout
#  Lag  4: 256  → 256   Linear + BN + ReLU + Dropout
#  Lag  5: 256  → 128   Linear + BN + ReLU + Dropout
#  Lag  6: 128  → 128   Linear + BN + ReLU + Dropout
#  Lag  7: 128  →  64   Linear + BN + ReLU + Dropout
#  Lag  8:  64  →  64   Linear + BN + ReLU + Dropout
#  Lag  9:  64  →  32   Linear + BN + ReLU + Dropout
#  Lag 10:  32  → n_classes  (output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
print(f"Classes ({n_classes}): {list(le.classes_)}")

DROP = 0.3

model = nn.Sequential(
    # Lag 1
    nn.Linear(13, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    # Lag 2
    nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    # Lag 3
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    # Lag 4
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    # Lag 5
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    # Lag 6
    nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    # Lag 7
    nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(DROP),
    # Lag 8
    nn.Linear(64, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(DROP),
    # Lag 9
    nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(DROP),
    # Lag 10 – output
    nn.Linear(32, n_classes),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# ── Train ─────────────────────────────────────────────────────────────────────

EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
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
    scheduler.step()
    print(f"Epoch {epoch:03d}/{EPOCHS}  loss: {total_loss/len(y_train):.4f}  "
          f"acc: {correct/len(y_train)*100:.1f}%")

# ── Evaluate ──────────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()

print("\n", classification_report(y_test, preds, target_names=list(le.classes_)))

# ── Save ──────────────────────────────────────────────────────────────────────

save_dir = os.path.join(_here, "..", "Models")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "emg_model_deep.pt")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {os.path.abspath(save_path)}")
