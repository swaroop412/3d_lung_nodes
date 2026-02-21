
import numpy as np
import sys
if "/tmp/pypackages" not in sys.path:
    sys.path.insert(0, "/tmp/pypackages")

# ── Configuration ─────────────────────────────────────────────────────────────
PATCH_SIZE   = (64, 64, 64)   # 3D patch shape (D, H, W) in voxels
HALF         = tuple(p // 2 for p in PATCH_SIZE)

# Label mapping: 0 = background, 1 = kidney, 2 = tumour
LABEL_NAMES  = {0: "background", 1: "kidney", 2: "tumour"}

# ── Find annotated (non-background) voxels ────────────────────────────────────
# Work on the resampled, normalised volume from the previous block
labeled_mask = (resampled_seg > 0)
foreground_coords = np.argwhere(labeled_mask)   # shape (N, 3)

print(f"Foreground voxels (label > 0): {foreground_coords.shape[0]:,}")
for label_id, label_name in LABEL_NAMES.items():
    if label_id == 0:
        continue
    count = int((resampled_seg == label_id).sum())
    print(f"  Label {label_id} ({label_name:>12}): {count:>10,} voxels")

# ── Sample patch centres from foreground voxels ──────────────────────────────
# Use every K-th foreground voxel so patches overlap minimally, then cap at 50
rng        = np.random.default_rng(42)
D, H, W    = resampled_volume.shape

# Filter coords that allow a full patch without boundary clipping
valid_mask = (
    (foreground_coords[:, 0] >= HALF[0]) & (foreground_coords[:, 0] < D - HALF[0]) &
    (foreground_coords[:, 1] >= HALF[1]) & (foreground_coords[:, 1] < H - HALF[1]) &
    (foreground_coords[:, 2] >= HALF[2]) & (foreground_coords[:, 2] < W - HALF[2])
)
valid_coords = foreground_coords[valid_mask]
print(f"\nValid patch-centre voxels (clearance ≥ {HALF}): {valid_coords.shape[0]:,}")

# Subsample up to 50 patches
n_patches   = min(50, valid_coords.shape[0])
chosen_idx  = rng.choice(valid_coords.shape[0], size=n_patches, replace=False)
centres     = valid_coords[chosen_idx]

print(f"Patches to extract          : {n_patches}")

# ── Extract patches and derive labels ────────────────────────────────────────
patches      = np.zeros((n_patches, *PATCH_SIZE), dtype=np.float32)
patch_segs   = np.zeros((n_patches, *PATCH_SIZE), dtype=np.int32)
patch_labels = np.zeros(n_patches, dtype=np.int32)

for i, (z, y, x) in enumerate(centres):
    patches[i]    = resampled_volume[z-HALF[0]:z+HALF[0],
                                     y-HALF[1]:y+HALF[1],
                                     x-HALF[2]:x+HALF[2]]
    patch_segs[i] = resampled_seg   [z-HALF[0]:z+HALF[0],
                                     y-HALF[1]:y+HALF[1],
                                     x-HALF[2]:x+HALF[2]]
    # Label = dominant non-background label in the patch seg
    seg_flat  = patch_segs[i].ravel()
    seg_flat  = seg_flat[seg_flat > 0]
    patch_labels[i] = int(np.bincount(seg_flat).argmax()) if len(seg_flat) else 0

# ── Patch statistics ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 3-D PATCH STATISTICS")
print(f"   Patch array shape  : {patches.shape}  (N, D, H, W)")
print(f"   Patch seg shape    : {patch_segs.shape}")
print(f"   Patch dtype        : {patches.dtype}")
print(f"   Patch value range  : [{patches.min():.4f}, {patches.max():.4f}]")
print(f"   Patch mean ± std   : {patches.mean():.4f} ± {patches.std():.4f}")
print()
print("   Label distribution across patches:")
for label_id in np.unique(patch_labels):
    n = int((patch_labels == label_id).sum())
    print(f"     Label {label_id} ({LABEL_NAMES.get(label_id,'?'):>12}): {n} patches")
print()
print("   Per-patch summary (first 5):")
print(f"   {'Idx':>4}  {'Centre (z,y,x)':>20}  {'Label':>8}  "
      f"{'Mean':>7}  {'Std':>7}  {'FG voxels':>10}")
for i in range(min(5, n_patches)):
    fg = int((patch_segs[i] > 0).sum())
    z, y, x = centres[i]
    print(f"   {i:>4}  ({z:>4},{y:>4},{x:>4})  "
          f"{LABEL_NAMES.get(patch_labels[i],'?'):>12}  "
          f"{patches[i].mean():>7.4f}  {patches[i].std():>7.4f}  {fg:>10,}")
