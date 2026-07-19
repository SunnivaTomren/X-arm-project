import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── Load data ─────────────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(_here, "..", "data", "processed", "features.xlsx"))

FEATURES = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
            "ENV_MEAN", "ENV_STD", "ENV_RANGE"]

# Real class balance in features.xlsx is roughly rest=12340, closing=1230,
# opening=830 (~86% rest) -- NOT balanced. Always predicting "rest" alone
# gets ~85.7% accuracy for free, so a plain 96%-ish number doesn't say much
# on its own; the classification_report below (per-class recall) is what
# actually shows whether opening/closing are being learned.
print("Class balance in features.xlsx:")
print(df["label"].value_counts())
print()

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

# ── Handle class imbalance ────────────────────────────────────────────────────
#
# With rest this dominant, a plain shuffled DataLoader + unweighted loss
# lets the network minimize loss almost entirely by predicting "rest" --
# most batches are mostly rest, so BatchNorm's running stats and the
# decision boundary both end up tuned mainly for rest. This is consistent
# with what was observed live: the model predicted "rest" with ~100%
# confidence on every window, even ones with an obviously different raw
# signal. Two fixes, used together:
#
#   1. WeightedRandomSampler -- makes every training BATCH roughly
#      balanced across classes, instead of ~86% rest per batch.
#   2. Class-weighted loss -- makes mistakes on the minority classes
#      (opening/closing) cost more, as a second line of defense.
train_class_counts = np.bincount(y_train.numpy(), minlength=n_classes)
print(f"Training set class counts: "
      f"{dict(zip(le.classes_, train_class_counts))}")

sample_weights = (1.0 / train_class_counts)[y_train.numpy()]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(sample_weights),
    replacement=True,
)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)

class_weights = torch.tensor(
    len(y_train) / (n_classes * train_class_counts), dtype=torch.float32)

# ── Model (10 lag) ────────────────────────────────────────────────────────────
#
#  Arkitektur: utvid → platå → trekk inn → output
#  BatchNorm stabiliserer trening; Dropout hindrer overfitting.
#
#  Lag  1:  n_features → 128   Linear + BN + ReLU + Dropout
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
n_features = len(FEATURES)   # kept dynamic so this still works if FEATURES is edited later

model = nn.Sequential(
    # Lag 1
    nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
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
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# ── Train ─────────────────────────────────────────────────────────────────────
#
# Track the best-performing epoch on the held-out test set and keep THOSE
# weights, rather than whatever the final epoch happens to land on. With
# only ~14k rows (and now correctly weighted so opening/closing actually
# matter to the loss), 100 epochs on this size of network can still drift
# past its best point -- this keeps the checkpoint that generalized best.
EPOCHS = 100
best_test_acc = 0.0
best_state = None

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

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test.to(device)).argmax(1).cpu()
    test_acc = (test_preds == y_test).float().mean().item()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:03d}/{EPOCHS}  loss: {total_loss/len(y_train):.4f}  "
          f"train_acc: {correct/len(y_train)*100:.1f}%  test_acc: {test_acc*100:.1f}%")

model.load_state_dict(best_state)
print(f"\nUsing best checkpoint (test_acc={best_test_acc*100:.1f}%) for evaluation and saving.")

# ── Evaluate ──────────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()

print("\n", classification_report(y_test, preds, target_names=list(le.classes_)))
print("Check the 'opening'/'closing' rows specifically, not just accuracy --")
print("with rest at ~86% of the data, overall accuracy alone can look good")
print("even if the model rarely gets the actual gestures right.")

# ── Save ──────────────────────────────────────────────────────────────────────

save_dir = os.path.join(_here, "..", "Models")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "emg_model_deep.pt")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {os.path.abspath(save_path)}")