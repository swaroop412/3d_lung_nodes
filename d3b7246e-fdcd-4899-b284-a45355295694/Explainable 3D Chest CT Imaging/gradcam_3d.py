
import subprocess, sys, os

# ── Install / locate PyTorch ─────────────────────────────────────────────────
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
if "/tmp/pypackages" not in sys.path:
    sys.path.insert(0, "/tmp/pypackages")

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print(f"PyTorch : {torch.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. RE-DEFINE ARCHITECTURE  (must match training block exactly)
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
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            _ResBlock3D(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            _ResBlock3D(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            _ResBlock3D(128),
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD SAVED WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_MODEL_PATH = "/tmp/lung_nodule_3d_resnet.pth"
assert os.path.exists(GRADCAM_MODEL_PATH), f"Model not found at {GRADCAM_MODEL_PATH}"

_gc_device = torch.device("cpu")
_gc_model  = _LungNoduleNet3D(num_classes=2).to(_gc_device)
_gc_model.load_state_dict(torch.load(GRADCAM_MODEL_PATH, map_location=_gc_device))
_gc_model.eval()
print(f"✓  Model loaded from {GRADCAM_MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAD-CAM HOOKS  (target = ResBlock3D inside layer3, index [3])
# ─────────────────────────────────────────────────────────────────────────────
_gc_activations = {}
_gc_gradients   = {}

def _fwd_hook_fn(module, inp, out):
    _gc_activations["layer3"] = out

def _bwd_hook_fn(module, grad_in, grad_out):
    _gc_gradients["layer3"] = grad_out[0]

_target = _gc_model.layer3[3]           # ResBlock3D(128) – last in layer3
_h_fwd  = _target.register_forward_hook(_fwd_hook_fn)
_h_bwd  = _target.register_full_backward_hook(_bwd_hook_fn)
print("✓  Hooks registered on layer3 → ResBlock3D(128)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECT FOREGROUND PATCH  (index 0 of `patches` = first kidney patch)
# ─────────────────────────────────────────────────────────────────────────────
_GC_IDX         = 0
_gc_patch_np    = patches[_GC_IDX].astype(np.float32)       # (64, 64, 64)
_gc_tensor      = torch.from_numpy(
    _gc_patch_np[None, None]                                 # (1,1,64,64,64)
).float()

print(f"✓  Patch #{_GC_IDX}  shape={_gc_patch_np.shape}  "
      f"val range=[{_gc_patch_np.min():.3f}, {_gc_patch_np.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FORWARD + BACKWARD FOR CLASS 1 (NODULE/FOREGROUND)
# ─────────────────────────────────────────────────────────────────────────────
_gc_model.zero_grad()
_gc_logits    = _gc_model(_gc_tensor)                        # (1, 2)
_gc_pred      = _gc_logits.argmax(dim=1).item()
_gc_score     = _gc_logits[0, 1]                             # class-1 score
_gc_score.backward()

print(f"✓  Forward done  —  predicted: {'foreground' if _gc_pred == 1 else 'background'}  "
      f"logits={_gc_logits[0].detach().numpy().round(3)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPUTE 3-D GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
_A      = _gc_activations["layer3"]               # (1, 128, d, h, w)
_G      = _gc_gradients["layer3"]                 # same shape

_alpha  = _G.mean(dim=(2, 3, 4), keepdim=True)    # (1, 128, 1, 1, 1)
_raw    = (_alpha * _A).sum(dim=1).squeeze()       # (d, h, w)
_raw    = torch.clamp(_raw, min=0.0)               # ReLU

_lo, _hi = _raw.min(), _raw.max()
if _hi > _lo:
    _raw_norm = ((_raw - _lo) / (_hi - _lo)).detach().numpy()
else:
    _raw_norm = np.zeros_like(_raw.detach().numpy())

print(f"✓  Raw CAM shape : {_raw_norm.shape}  "
      f"range=[{_raw_norm.min():.3f}, {_raw_norm.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 7. UPSAMPLE TO PATCH SIZE 64×64×64 (trilinear, order=1)
# ─────────────────────────────────────────────────────────────────────────────
_zf = tuple(
    _gc_patch_np.shape[k] / _raw_norm.shape[k] for k in range(3)
)
gradcam_heatmap_3d = np.clip(zoom(_raw_norm, _zf, order=1), 0, 1)  # (64,64,64)

print(f"✓  Upsampled heatmap : {gradcam_heatmap_3d.shape}  "
      f"zoom={tuple(round(f,1) for f in _zf)}")

_h_fwd.remove()
_h_bwd.remove()

# ─────────────────────────────────────────────────────────────────────────────
# 8. PEAK ACTIVATION COORDINATES
# ─────────────────────────────────────────────────────────────────────────────
gradcam_peak_coords = np.unravel_index(
    np.argmax(gradcam_heatmap_3d), gradcam_heatmap_3d.shape
)
gradcam_peak_value  = float(gradcam_heatmap_3d[gradcam_peak_coords])

print(f"\n{'═'*55}")
print(f"  ★  Peak Grad-CAM activation")
print(f"     Voxel (z, y, x) : {gradcam_peak_coords}")
print(f"     Value            : {gradcam_peak_value:.4f}")
print(f"{'═'*55}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALISE 3 AXIAL SLICES  (Zerve dark theme)
# ─────────────────────────────────────────────────────────────────────────────
_BG   = "#1D1D20"
_TXT  = "#fbfbff"
_SEC  = "#909094"

_nz      = _gc_patch_np.shape[0]       # 64
_slice_zs = [_nz // 4, _nz // 2, 3 * _nz // 4]   # [16, 32, 48]

# Zerve-styled transparent → purple → orange → gold colormap
_gc_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gc_zerve",
    [
        (0.00, (0.000, 0.000, 0.000, 0.00)),
        (0.30, (0.250, 0.000, 0.500, 0.60)),
        (0.60, (1.000, 0.500, 0.000, 0.80)),
        (1.00, (1.000, 0.843, 0.000, 1.00)),
    ]
)

# ── Slice 1 ──────────────────────────────────────────────────────────────────
_sz0 = _slice_zs[0]
gradcam_fig_slice0, _axs = plt.subplots(1, 3, figsize=(15, 5))
gradcam_fig_slice0.patch.set_facecolor(_BG)
gradcam_fig_slice0.suptitle(
    f"3D Grad-CAM  ·  Axial z={_sz0}  ·  Pred: {'Foreground ✓' if _gc_pred == 1 else 'Background ✗'}",
    color=_TXT, fontsize=13, fontweight="bold", y=1.01
)
for _ax in _axs:
    _ax.set_facecolor(_BG)
_axs[0].imshow(_gc_patch_np[_sz0], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[0].set_title("CT Patch", color=_TXT, fontsize=11); _axs[0].axis("off")
_im_h0 = _axs[1].imshow(gradcam_heatmap_3d[_sz0], cmap="hot", vmin=0, vmax=1, origin="lower")
_axs[1].set_title("Grad-CAM Heatmap", color=_TXT, fontsize=11); _axs[1].axis("off")
_cb = gradcam_fig_slice0.colorbar(_im_h0, ax=_axs[1], fraction=0.046, pad=0.04)
_cb.ax.tick_params(colors=_SEC, labelsize=8); _cb.outline.set_edgecolor(_SEC)
_axs[2].imshow(_gc_patch_np[_sz0], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[2].imshow(gradcam_heatmap_3d[_sz0], cmap=_gc_cmap, vmin=0, vmax=1, alpha=0.65, origin="lower")
if _sz0 == gradcam_peak_coords[0]:
    _axs[2].scatter(gradcam_peak_coords[2], gradcam_peak_coords[1],
                    c="#ffd400", s=140, marker="*", edgecolors=_BG, zorder=5, label="Peak")
    _axs[2].legend(facecolor="#2a2a2e", edgecolor=_SEC, labelcolor=_TXT, fontsize=9)
_axs[2].set_title("CT + Grad-CAM Overlay", color=_TXT, fontsize=11); _axs[2].axis("off")
plt.tight_layout()

# ── Slice 2 ──────────────────────────────────────────────────────────────────
_sz1 = _slice_zs[1]
gradcam_fig_slice1, _axs = plt.subplots(1, 3, figsize=(15, 5))
gradcam_fig_slice1.patch.set_facecolor(_BG)
gradcam_fig_slice1.suptitle(
    f"3D Grad-CAM  ·  Axial z={_sz1}  ·  Pred: {'Foreground ✓' if _gc_pred == 1 else 'Background ✗'}",
    color=_TXT, fontsize=13, fontweight="bold", y=1.01
)
for _ax in _axs:
    _ax.set_facecolor(_BG)
_axs[0].imshow(_gc_patch_np[_sz1], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[0].set_title("CT Patch", color=_TXT, fontsize=11); _axs[0].axis("off")
_im_h1 = _axs[1].imshow(gradcam_heatmap_3d[_sz1], cmap="hot", vmin=0, vmax=1, origin="lower")
_axs[1].set_title("Grad-CAM Heatmap", color=_TXT, fontsize=11); _axs[1].axis("off")
_cb = gradcam_fig_slice1.colorbar(_im_h1, ax=_axs[1], fraction=0.046, pad=0.04)
_cb.ax.tick_params(colors=_SEC, labelsize=8); _cb.outline.set_edgecolor(_SEC)
_axs[2].imshow(_gc_patch_np[_sz1], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[2].imshow(gradcam_heatmap_3d[_sz1], cmap=_gc_cmap, vmin=0, vmax=1, alpha=0.65, origin="lower")
if _sz1 == gradcam_peak_coords[0]:
    _axs[2].scatter(gradcam_peak_coords[2], gradcam_peak_coords[1],
                    c="#ffd400", s=140, marker="*", edgecolors=_BG, zorder=5, label="Peak")
    _axs[2].legend(facecolor="#2a2a2e", edgecolor=_SEC, labelcolor=_TXT, fontsize=9)
_axs[2].set_title("CT + Grad-CAM Overlay", color=_TXT, fontsize=11); _axs[2].axis("off")
plt.tight_layout()

# ── Slice 3 ──────────────────────────────────────────────────────────────────
_sz2 = _slice_zs[2]
gradcam_fig_slice2, _axs = plt.subplots(1, 3, figsize=(15, 5))
gradcam_fig_slice2.patch.set_facecolor(_BG)
gradcam_fig_slice2.suptitle(
    f"3D Grad-CAM  ·  Axial z={_sz2}  ·  Pred: {'Foreground ✓' if _gc_pred == 1 else 'Background ✗'}",
    color=_TXT, fontsize=13, fontweight="bold", y=1.01
)
for _ax in _axs:
    _ax.set_facecolor(_BG)
_axs[0].imshow(_gc_patch_np[_sz2], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[0].set_title("CT Patch", color=_TXT, fontsize=11); _axs[0].axis("off")
_im_h2 = _axs[2].imshow(gradcam_heatmap_3d[_sz2], cmap="hot", vmin=0, vmax=1, origin="lower")
_axs[1].imshow(gradcam_heatmap_3d[_sz2], cmap="hot", vmin=0, vmax=1, origin="lower")
_axs[1].set_title("Grad-CAM Heatmap", color=_TXT, fontsize=11); _axs[1].axis("off")
_cb = gradcam_fig_slice2.colorbar(_im_h2, ax=_axs[1], fraction=0.046, pad=0.04)
_cb.ax.tick_params(colors=_SEC, labelsize=8); _cb.outline.set_edgecolor(_SEC)
_axs[2].imshow(_gc_patch_np[_sz2], cmap="gray", vmin=0, vmax=1, origin="lower")
_axs[2].imshow(gradcam_heatmap_3d[_sz2], cmap=_gc_cmap, vmin=0, vmax=1, alpha=0.65, origin="lower")
if _sz2 == gradcam_peak_coords[0]:
    _axs[2].scatter(gradcam_peak_coords[2], gradcam_peak_coords[1],
                    c="#ffd400", s=140, marker="*", edgecolors=_BG, zorder=5, label="Peak")
    _axs[2].legend(facecolor="#2a2a2e", edgecolor=_SEC, labelcolor=_TXT, fontsize=9)
_axs[2].set_title("CT + Grad-CAM Overlay", color=_TXT, fontsize=11); _axs[2].axis("off")
plt.tight_layout()

print(f"✓  3 axial Grad-CAM figures generated  (z = {_slice_zs})")
print(f"   ★  Peak activation voxel  →  z={gradcam_peak_coords[0]}, "
      f"y={gradcam_peak_coords[1]}, x={gradcam_peak_coords[2]}  "
      f"(value={gradcam_peak_value:.4f})")
