
import subprocess, sys, os

# ── Install PyTorch (CPU-only) ─────────────────────────────────────────────
_install = subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--target=/tmp/pypackages", "--quiet",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ],
    capture_output=True, text=True
)
print("torch install returncode:", _install.returncode)
if _install.returncode != 0:
    print(_install.stderr[-500:])

if "/tmp/pypackages" not in sys.path:
    sys.path.insert(0, "/tmp/pypackages")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, f1_score
)

print(f"PyTorch {torch.__version__} | scikit-learn ready")

# ─────────────────────────────────────────────────────────────────────────────
# Zerve dark-theme palette
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#1D1D20"
TXT    = "#fbfbff"
SEC    = "#909094"
C_ROC  = "#A1C9F4"
C_DIAG = "#ffd400"
C_GRID = "#2a2a2e"

# ─────────────────────────────────────────────────────────────────────────────
# 1. RE-BUILD ARCHITECTURE  (identical to train block)
# ─────────────────────────────────────────────────────────────────────────────
class _ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.net(x) + x)


class _LungNoduleNet3D(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True), _ResBlock3D(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True), _ResBlock3D(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True), _ResBlock3D(128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))))


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET  &  IDENTICAL TRAIN/VAL SPLIT  (seed=42 → same as training block)
# ─────────────────────────────────────────────────────────────────────────────
class _PatchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X[:, None])   # (N, 1, D, H, W)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

torch.manual_seed(42)
np.random.seed(42)
EVAL_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_full_ds = _PatchDS(all_patches, all_labels)
_n_total = len(_full_ds)
_n_val   = max(1, int(0.2 * _n_total))
_n_train = _n_total - _n_val

_train_sub, _val_sub = torch.utils.data.random_split(
    _full_ds, [_n_train, _n_val],
    generator=torch.Generator().manual_seed(42)
)
_train_loader = DataLoader(_train_sub, batch_size=4, shuffle=True,
                           generator=torch.Generator().manual_seed(42))
_val_loader   = DataLoader(_val_sub,   batch_size=4, shuffle=False)
print(f"Dataset  →  train={_n_train}  val={_n_val}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RE-TRAIN  (10 epochs, exact same hyper-params as train block)
# ─────────────────────────────────────────────────────────────────────────────
_eval_model = _LungNoduleNet3D(num_classes=2).to(EVAL_DEV)
_crit       = nn.CrossEntropyLoss()
_opt        = optim.Adam(_eval_model.parameters(), lr=1e-3, weight_decay=1e-4)
_sched      = optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=10, eta_min=1e-5)

print("\nRe-training 10 epochs for evaluation …")
for _ep in range(1, 11):
    _eval_model.train()
    for _xb, _yb in _train_loader:
        _xb, _yb = _xb.to(EVAL_DEV), _yb.to(EVAL_DEV)
        _opt.zero_grad()
        _loss = _crit(_eval_model(_xb), _yb)
        _loss.backward()
        _opt.step()
    _sched.step()
print("Re-training complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────
_eval_model.eval()
eval_y_true, eval_y_pred, eval_y_prob = [], [], []

with torch.no_grad():
    for _xb, _yb in _val_loader:
        _xb = _xb.to(EVAL_DEV)
        _logits = _eval_model(_xb)
        _probs  = torch.softmax(_logits, dim=1)[:, 1]
        _preds  = _logits.argmax(dim=1)
        eval_y_true.extend(_yb.numpy().tolist())
        eval_y_pred.extend(_preds.cpu().numpy().tolist())
        eval_y_prob.extend(_probs.cpu().numpy().tolist())

eval_y_true = np.array(eval_y_true)
eval_y_pred = np.array(eval_y_pred)
eval_y_prob = np.array(eval_y_prob)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CLINICAL / MEDICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
TP = int(((eval_y_pred == 1) & (eval_y_true == 1)).sum())
TN = int(((eval_y_pred == 0) & (eval_y_true == 0)).sum())
FP = int(((eval_y_pred == 1) & (eval_y_true == 0)).sum())
FN = int(((eval_y_pred == 0) & (eval_y_true == 1)).sum())

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
precision   = precision_score(eval_y_true, eval_y_pred, zero_division=0)
f1          = f1_score(eval_y_true, eval_y_pred, zero_division=0)
auc_roc     = roc_auc_score(eval_y_true, eval_y_prob)
fpr, tpr, thresholds = roc_curve(eval_y_true, eval_y_prob)

# Youden's J  →  optimal threshold
_j_idx     = np.argmax(tpr - fpr)
opt_thresh = float(thresholds[_j_idx])
opt_sens   = float(tpr[_j_idx])
opt_spec   = float(1 - fpr[_j_idx])

print("═" * 55)
print("  CLINICAL EVALUATION METRICS  —  Validation Set")
print("═" * 55)
print(f"  Sensitivity  (Recall / TPR) : {sensitivity:.4f}  ({sensitivity*100:.2f}%)")
print(f"  Specificity  (TNR)          : {specificity:.4f}  ({specificity*100:.2f}%)")
print(f"  Precision    (PPV)          : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  F1-Score                    : {f1:.4f}")
print(f"  AUC-ROC                     : {auc_roc:.4f}")
print(f"  ─── Optimal threshold (Youden) : {opt_thresh:.4f}")
print(f"      Sensitivity @ opt       : {opt_sens:.4f}  ({opt_sens*100:.2f}%)")
print(f"      Specificity @ opt       : {opt_spec:.4f}  ({opt_spec*100:.2f}%)")
print("─" * 55)
print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print("═" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL SKLEARN CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n──── Full Classification Report ────────────────────────")
print(classification_report(
    eval_y_true, eval_y_pred,
    target_names=["Background (0)", "Foreground (1)"],
    digits=4
))

# ─────────────────────────────────────────────────────────────────────────────
# 7. ROC CURVE  (Zerve dark theme)
# ─────────────────────────────────────────────────────────────────────────────
fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
fig_roc.patch.set_facecolor(BG)
ax_roc.set_facecolor(BG)

ax_roc.plot(fpr, tpr, color=C_ROC, lw=2.5,
            label=f"3D ResNet  AUC = {auc_roc:.4f}")
ax_roc.plot([0, 1], [0, 1], color=C_DIAG, lw=1.5, linestyle="--",
            label="Random classifier")
ax_roc.scatter([fpr[_j_idx]], [tpr[_j_idx]], color="#17b26a",
               s=140, zorder=5, label=f"Optimal threshold ({opt_thresh:.3f})")
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=C_ROC)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.05)
ax_roc.set_title("ROC Curve — 3D ResNet Kidney Lesion Classifier",
                 color=TXT, fontsize=14, fontweight="bold", pad=14)
ax_roc.set_xlabel("False Positive Rate  (1 − Specificity)", color=SEC, fontsize=12)
ax_roc.set_ylabel("True Positive Rate  (Sensitivity)", color=SEC, fontsize=12)
ax_roc.tick_params(colors=SEC, labelsize=10)
for _sp in ax_roc.spines.values():
    _sp.set_color(SEC)
ax_roc.legend(facecolor=C_GRID, edgecolor=SEC, labelcolor=TXT, fontsize=11)
ax_roc.set_aspect("equal")
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# 8. CONFUSION MATRIX  (Zerve dark theme)
# ─────────────────────────────────────────────────────────────────────────────
eval_cm = confusion_matrix(eval_y_true, eval_y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
fig_cm.patch.set_facecolor(BG)
ax_cm.set_facecolor(BG)

_im = ax_cm.imshow(eval_cm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=eval_cm.max() + 1)

_thresh_cm  = eval_cm.max() / 2.0
_cell_types = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
for _r in range(2):
    for _c in range(2):
        _val   = eval_cm[_r, _c]
        _label = f"{_cell_types[(_r, _c)]}\n{_val}"
        _color = TXT if _val < _thresh_cm else BG
        ax_cm.text(_c, _r, _label, ha="center", va="center",
                   fontsize=16, fontweight="bold", color=_color)

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Background (0)", "Foreground (1)"], color=SEC, fontsize=10)
ax_cm.set_yticklabels(["Background (0)", "Foreground (1)"], color=SEC, fontsize=10)
ax_cm.set_xlabel("Predicted Label", color=SEC, fontsize=12, labelpad=10)
ax_cm.set_ylabel("True Label",      color=SEC, fontsize=12, labelpad=10)
ax_cm.set_title("Confusion Matrix — Validation Set",
                color=TXT, fontsize=14, fontweight="bold", pad=14)
ax_cm.tick_params(colors=SEC)
for _sp in ax_cm.spines.values():
    _sp.set_color(SEC)
_cbar = fig_cm.colorbar(_im, ax=ax_cm, fraction=0.046, pad=0.04)
_cbar.ax.yaxis.set_tick_params(color=SEC)
_cbar.ax.tick_params(labelcolor=SEC)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT KEY METRICS DICT
# ─────────────────────────────────────────────────────────────────────────────
eval_metrics = {
    "sensitivity":       sensitivity,
    "specificity":       specificity,
    "precision":         precision,
    "f1_score":          f1,
    "auc_roc":           auc_roc,
    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    "optimal_threshold": opt_thresh,
    "n_val_samples":     int(_n_val),
}

print("\n✓  ROC curve and confusion matrix generated.")
print(f"✓  eval_metrics dict exported ({len(eval_metrics)} keys).")
