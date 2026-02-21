import torch

# --- Model & Training ---
MODEL_PATH = "lung_nodule_net.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # e.g., 0: Non-Nodule, 1: Nodule
LABELS = {0: "Non-Nodule", 1: "Nodule"}

# --- Preprocessing ---
PATCH_SIZE = (64, 64, 64)  # The input size for the 3D CNN (Depth, Height, Width)
HU_MIN, HU_MAX = -1000, 400  # Hounsfield Unit windowing for lung tissue