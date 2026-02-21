
import subprocess, sys, os

# ── Install PyTorch (CPU-only) into /tmp/pypackages ──────────────────────────
_install = subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--target=/tmp/pypackages",
        "--quiet",
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
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"PyTorch version : {torch.__version__}")

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD A BALANCED BINARY DATASET
#    Foreground patches  →  label 1 (kidney / nodule)
#    Background patches  →  label 0 (sampled where resampled_seg == 0)
# ─────────────────────────────────────────────────────────────────────────────
HALF_S = tuple(p // 2 for p in PATCH_SIZE)   # (32,32,32)
rng_bg = np.random.default_rng(99)

bg_mask   = (resampled_seg == 0)
bg_coords = np.argwhere(bg_mask)
D_v, H_v, W_v = resampled_volume.shape

border_ok = (
    (bg_coords[:, 0] >= HALF_S[0]) & (bg_coords[:, 0] < D_v - HALF_S[0]) &
    (bg_coords[:, 1] >= HALF_S[1]) & (bg_coords[:, 1] < H_v - HALF_S[1]) &
    (bg_coords[:, 2] >= HALF_S[2]) & (bg_coords[:, 2] < W_v - HALF_S[2])
)
bg_valid  = bg_coords[border_ok]
n_bg      = min(n_patches, bg_valid.shape[0])
bg_idx    = rng_bg.choice(bg_valid.shape[0], size=n_bg, replace=False)
bg_centres = bg_valid[bg_idx]

bg_patches = np.zeros((n_bg, *PATCH_SIZE), dtype=np.float32)
for _i, (_z, _y, _x) in enumerate(bg_centres):
    bg_patches[_i] = resampled_volume[
        _z - HALF_S[0]: _z + HALF_S[0],
        _y - HALF_S[1]: _y + HALF_S[1],
        _x - HALF_S[2]: _x + HALF_S[2],
    ]

all_patches = np.concatenate([patches, bg_patches], axis=0)
all_labels  = np.concatenate([
    np.ones(n_patches, dtype=np.int64),
    np.zeros(n_bg,     dtype=np.int64)
], axis=0)

print(f"\nTotal patches  : {len(all_patches)}  "
      f"(foreground={n_patches}, background={n_bg})")
print(f"Patch shape    : {all_patches.shape}   dtype={all_patches.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """Adds a channel dimension → (N, 1, D, H, W)."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X[:, None, :, :, :])
        self.y = torch.from_numpy(y)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


full_ds = PatchDataset(all_patches, all_labels)
n_total = len(full_ds)
n_val   = max(1, int(0.2 * n_total))
n_train = n_total - n_val

train_ds, val_ds = random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

BATCH        = 4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
print(f"Train samples  : {n_train}  |  Val samples : {n_val}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 3D ResNet-STYLE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    """Two 3×3×3 convolutions with a skip connection."""
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class LungNoduleNet3D(nn.Module):
    """
    Lightweight 3-D ResNet for binary patch classification.
    Input  : (B, 1, 64, 64, 64)
    Output : (B, 2)  —  logits for [background, foreground]
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem   = nn.Sequential(                               # 64 → 16
            nn.Conv3d(1,  16, 7, stride=2, padding=3, bias=False), # 64→32
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),                  # 32→16
        )
        self.layer1 = nn.Sequential(                               # 16 → 8
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            ResBlock3D(32),
        )
        self.layer2 = nn.Sequential(                               # 8 → 4
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64),
        )
        self.layer3 = nn.Sequential(                               # 4 → 2
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            ResBlock3D(128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


model        = LungNoduleNet3D(num_classes=2).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel : LungNoduleNet3D  |  Trainable params : {total_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

N_EPOCHS    = 10
cnn_train_losses, cnn_val_losses = [], []
cnn_train_accs,   cnn_val_accs   = [], []

print("\n" + "─" * 65)
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>9}  {'Val Acc':>8}")
print("─" * 65)

for _epoch in range(1, N_EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────────────────
    model.train()
    _t_loss, _t_correct, _t_total = 0.0, 0, 0
    for _xb, _yb in train_loader:
        _xb, _yb = _xb.to(DEVICE), _yb.to(DEVICE)
        optimizer.zero_grad()
        _logits = model(_xb)
        _loss   = criterion(_logits, _yb)
        _loss.backward()
        optimizer.step()
        _t_loss    += _loss.item() * len(_yb)
        _t_correct += (_logits.argmax(1) == _yb).sum().item()
        _t_total   += len(_yb)

    _train_loss = _t_loss / _t_total
    _train_acc  = _t_correct / _t_total

    # ── Validate ──────────────────────────────────────────────────────────
    model.eval()
    _v_loss, _v_correct, _v_total = 0.0, 0, 0
    with torch.no_grad():
        for _xb, _yb in val_loader:
            _xb, _yb = _xb.to(DEVICE), _yb.to(DEVICE)
            _logits  = model(_xb)
            _v_loss    += criterion(_logits, _yb).item() * len(_yb)
            _v_correct += (_logits.argmax(1) == _yb).sum().item()
            _v_total   += len(_yb)

    _val_loss = _v_loss / _v_total
    _val_acc  = _v_correct / _v_total

    cnn_train_losses.append(_train_loss)
    cnn_val_losses.append(_val_loss)
    cnn_train_accs.append(_train_acc)
    cnn_val_accs.append(_val_acc)

    scheduler.step()
    print(f"{_epoch:>5}  {_train_loss:>10.4f}  {_train_acc:>8.2%}  "
          f"{_val_loss:>9.4f}  {_val_acc:>7.2%}")

print("─" * 65)
print(f"\n✓  Final Train acc  : {cnn_train_accs[-1]:.2%}")
print(f"✓  Final Val   acc  : {cnn_val_accs[-1]:.2%}")
print(f"✓  Final Train loss : {cnn_train_losses[-1]:.4f}")
print(f"✓  Final Val   loss : {cnn_val_losses[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE MODEL WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "/tmp/lung_nodule_3d_resnet.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel weights saved → {MODEL_PATH}  "
      f"({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOSS & ACCURACY CURVES (Zerve dark theme)
# ─────────────────────────────────────────────────────────────────────────────
BG_C    = "#1D1D20"
TXT_C   = "#fbfbff"
SEC_C   = "#909094"
C_TRAIN = "#A1C9F4"
C_VAL   = "#FFB482"
ep_ax   = list(range(1, N_EPOCHS + 1))

# ── Loss curve ────────────────────────────────────────────────────────────────
fig_loss, ax_loss = plt.subplots(figsize=(9, 5))
fig_loss.patch.set_facecolor(BG_C)
ax_loss.set_facecolor(BG_C)
ax_loss.plot(ep_ax, cnn_train_losses, color=C_TRAIN, lw=2.5,
             marker="o", markersize=5, label="Train Loss")
ax_loss.plot(ep_ax, cnn_val_losses,   color=C_VAL,   lw=2.5,
             marker="s", markersize=5, linestyle="--", label="Val Loss")
ax_loss.set_title("3D ResNet – Cross-Entropy Loss per Epoch",
                  color=TXT_C, fontsize=14, fontweight="bold", pad=12)
ax_loss.set_xlabel("Epoch", color=SEC_C, fontsize=11)
ax_loss.set_ylabel("Loss",  color=SEC_C, fontsize=11)
ax_loss.tick_params(colors=SEC_C)
for spine in ax_loss.spines.values():
    spine.set_color(SEC_C)
ax_loss.set_xticks(ep_ax)
ax_loss.legend(facecolor="#2a2a2e", edgecolor=SEC_C,
               labelcolor=TXT_C, fontsize=10)
plt.tight_layout()

# ── Accuracy curve ────────────────────────────────────────────────────────────
fig_acc, ax_acc = plt.subplots(figsize=(9, 5))
fig_acc.patch.set_facecolor(BG_C)
ax_acc.set_facecolor(BG_C)
ax_acc.plot(ep_ax, [a * 100 for a in cnn_train_accs], color=C_TRAIN, lw=2.5,
            marker="o", markersize=5, label="Train Acc")
ax_acc.plot(ep_ax, [a * 100 for a in cnn_val_accs],   color=C_VAL, lw=2.5,
            marker="s", markersize=5, linestyle="--", label="Val Acc")
ax_acc.set_title("3D ResNet – Accuracy per Epoch",
                 color=TXT_C, fontsize=14, fontweight="bold", pad=12)
ax_acc.set_xlabel("Epoch", color=SEC_C, fontsize=11)
ax_acc.set_ylabel("Accuracy (%)", color=SEC_C, fontsize=11)
ax_acc.set_ylim(0, 105)
ax_acc.tick_params(colors=SEC_C)
for spine in ax_acc.spines.values():
    spine.set_color(SEC_C)
ax_acc.set_xticks(ep_ax)
ax_acc.legend(facecolor="#2a2a2e", edgecolor=SEC_C,
              labelcolor=TXT_C, fontsize=10)
plt.tight_layout()
