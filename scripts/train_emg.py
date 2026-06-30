"""
EMG Gesture Classifier med PyTorch
====================================
Les features.xlsx -> tren et nevralt nettverk -> evaluer med noyaktighet og confusion matrix
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "..", "output")

# -----------------------------------------------------------------------
# 1. LAST INN DATA
# -----------------------------------------------------------------------
df = pd.read_excel(os.path.join(OUTPUT_DIR, "features.xlsx"))

FEATURE_KOLS = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
                "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]

X = df[FEATURE_KOLS].values.astype(np.float32)
y_tekst = df["label"].values

print("Klasser funnet:", np.unique(y_tekst))
print("Fordeling:")
print(pd.Series(y_tekst).value_counts())
print(f"\nAntall vinduer totalt: {len(X)}, antall features: {X.shape[1]}")

# -----------------------------------------------------------------------
# 2. KODE LABELS SOM TALL  (rest=0, opening=1, closing=2 e.l.)
# -----------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y_tekst).astype(np.int64)
KLASSER = le.classes_
N_KLASSER = len(KLASSER)
print(f"\nLabel-mapping: {dict(zip(KLASSER, le.transform(KLASSER)))}")

# -----------------------------------------------------------------------
# 3. NORMALISER FEATURES  (viktig for nevrale nettverk)
# -----------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# -----------------------------------------------------------------------
# 4. TREN/TEST-SPLIT  (80% trening, 20% test)
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreningsett: {len(X_train)} vinduer | Testsett: {len(X_test)} vinduer")

# Pakk inn i PyTorch TensorDataset og DataLoader
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# -----------------------------------------------------------------------
# 5. DEFINER MODELLEN  (MLP: Multi-Layer Perceptron)
# -----------------------------------------------------------------------
class EMGClassifier(nn.Module):
    def __init__(self, n_features, n_klasser):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_klasser),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nBruker: {device}")

model     = EMGClassifier(len(FEATURE_KOLS), N_KLASSER).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# -----------------------------------------------------------------------
# 6. TRENINGSLØKKE
# -----------------------------------------------------------------------
EPOKER = 60

tap_historikk = []
print("\n--- TRENING STARTER ---")

for epoke in range(1, EPOKER + 1):
    model.train()
    totalt_tap = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        tap  = criterion(pred, y_batch)
        tap.backward()
        optimizer.step()
        totalt_tap += tap.item() * len(X_batch)

    scheduler.step()
    snitt_tap = totalt_tap / len(train_ds)
    tap_historikk.append(snitt_tap)

    if epoke % 10 == 0 or epoke == 1:
        print(f"  Epoke {epoke:3d}/{EPOKER}  tap={snitt_tap:.4f}")

# -----------------------------------------------------------------------
# 7. EVALUER PÅ TESTSETTET
# -----------------------------------------------------------------------
model.eval()
alle_pred, alle_fasit = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).argmax(dim=1).cpu().numpy()
        alle_pred.extend(pred)
        alle_fasit.extend(y_batch.numpy())

alle_pred  = np.array(alle_pred)
alle_fasit = np.array(alle_fasit)
noyaktighet = (alle_pred == alle_fasit).mean() * 100
print(f"\n=== NØYAKTIGHET PÅ TESTSETTET: {noyaktighet:.1f}% ===")

print("\nDetaljert rapport:")
print(classification_report(alle_fasit, alle_pred, target_names=KLASSER))

# -----------------------------------------------------------------------
# 8. PLOT: TAP OG CONFUSION MATRIX
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Tap-kurve
ax1.plot(range(1, EPOKER + 1), tap_historikk, color="steelblue")
ax1.set_xlabel("Epoke")
ax1.set_ylabel("Tap (CrossEntropy)")
ax1.set_title("Trenings-tap over tid")
ax1.grid(True)

# Confusion matrix
cm = confusion_matrix(alle_fasit, alle_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=KLASSER, yticklabels=KLASSER, ax=ax2)
ax2.set_xlabel("Predikert")
ax2.set_ylabel("Fasit")
ax2.set_title("Confusion Matrix (testsett)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "resultat_trening.png"), dpi=120)
plt.show()
print(f"\nPlot lagret som {os.path.join(OUTPUT_DIR, 'resultat_trening.png')}")

# -----------------------------------------------------------------------
# 9. LAGRE MODELLEN
# -----------------------------------------------------------------------
torch.save({
    "model_state": model.state_dict(),
    "klasser":     KLASSER.tolist(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std":  scaler.scale_.tolist(),
}, os.path.join(OUTPUT_DIR, "emg_model.pt"))
print(f"Modell lagret som {os.path.join(OUTPUT_DIR, 'emg_model.pt')}")
