
import subprocess, sys

# Install nibabel to /tmp which is writable in this Lambda/Fargate env
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--target=/tmp/pypackages", "nibabel"],
    capture_output=True, text=True
)
print("returncode:", result.returncode)
print("stderr:", result.stderr[-300:] if result.stderr else "(none)")

# Prepend the target dir so Python can find it
import importlib
if "/tmp/pypackages" not in sys.path:
    sys.path.insert(0, "/tmp/pypackages")

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

print("nibabel version:", nib.__version__)

# ── 1. Load NIfTI files ──────────────────────────────────────────────────────
imaging_nii   = nib.load("imaging.nii.gz")
seg_nii       = nib.load("segmentation.nii.gz")

imaging_volume = imaging_nii.get_fdata(dtype=np.float32)
seg_volume     = seg_nii.get_fdata(dtype=np.float32).astype(np.int32)

orig_spacing = tuple(
    float(np.abs(imaging_nii.header.get_zooms()[i])) for i in range(3)
)

print("=" * 60)
print("1. RAW VOLUME INFO")
print(f"   Imaging shape    : {imaging_volume.shape}")
print(f"   Seg shape        : {seg_volume.shape}")
print(f"   Original spacing : {orig_spacing} mm")
print(f"   HU range (raw)   : [{imaging_volume.min():.1f}, {imaging_volume.max():.1f}]")
print(f"   Unique seg labels: {np.unique(seg_volume).tolist()}")

# ── 2. HU Windowing (Lung window: −1000 to +400 HU) ─────────────────────────
HU_MIN, HU_MAX = -1000.0, 400.0
windowed_volume = np.clip(imaging_volume, HU_MIN, HU_MAX)

print(f"\n2. HU WINDOWING  (Lung window: {int(HU_MIN)} – {int(HU_MAX)} HU)")
print(f"   HU range (windowed): [{windowed_volume.min():.1f}, {windowed_volume.max():.1f}]")

# ── 3. Normalise to [0, 1] ────────────────────────────────────────────────────
normalized_volume = (windowed_volume - HU_MIN) / (HU_MAX - HU_MIN)

print("\n3. NORMALISATION  (min–max to [0, 1])")
print(f"   Value range (normalised): [{normalized_volume.min():.4f}, {normalized_volume.max():.4f}]")
print(f"   Mean : {normalized_volume.mean():.4f}  |  Std: {normalized_volume.std():.4f}")

# ── 4. Resample to fixed voxel spacing (1×1×1 mm) ────────────────────────────
TARGET_SPACING = (1.0, 1.0, 1.0)
zoom_factors   = tuple(orig_spacing[i] / TARGET_SPACING[i] for i in range(3))

resampled_volume = zoom(normalized_volume, zoom_factors, order=1)
resampled_seg    = zoom(seg_volume.astype(np.float32), zoom_factors, order=0).astype(np.int32)

print(f"\n4. RESAMPLING  (target: {TARGET_SPACING} mm)")
print(f"   Zoom factors               : {tuple(round(z, 4) for z in zoom_factors)}")
print(f"   Resampled volume shape     : {resampled_volume.shape}")
print(f"   Resampled seg shape        : {resampled_seg.shape}")
print(f"   Effective spacing (mm)     : {TARGET_SPACING}")
print(f"   Unique resampled seg labels: {np.unique(resampled_seg).tolist()}")
