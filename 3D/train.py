import os
import glob
# The following line is a workaround for a common issue on Windows with Anaconda,
# where multiple OpenMP runtimes can be loaded, causing a crash.
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import LungNoduleNet3D
from config import MODEL_PATH, DEVICE, NUM_CLASSES
from utils import preprocess_nifti, centre_patch

# --- Configuration ---
DATA_DIR = "data"
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCHS = 5

class NoduleDataset(Dataset):
    def __init__(self, data_dir):
        # Find all NIfTI files recursively
        self.files = glob.glob(os.path.join(data_dir, "**", "*.nii*"), recursive=True)
        if not self.files:
            print(f"WARNING: No .nii or .nii.gz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # 1. Load and Preprocess
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        vol, _ = preprocess_nifti(file_bytes)
        patch = centre_patch(vol)
        
        # 2. Convert to Tensor (Add Channel Dimension: [1, D, H, W])
        tensor = torch.from_numpy(patch).float().unsqueeze(0)
        
        # 3. Determine Label from Filename
        # Heuristic: If filename contains "nodule", label=1, else 0
        filename = os.path.basename(file_path).lower()
        label = 1 if "nodule" in filename else 0
        
        return tensor, torch.tensor(label, dtype=torch.long)

def train():
    print(f"--- Starting Training on device: {DEVICE} ---")
    
    # 1. Setup Data
    dataset = NoduleDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No data found. Please create a 'data' folder and add .nii files.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Found {len(dataset)} scans.")

    # 2. Setup Model
    model = LungNoduleNet3D(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(dataloader):.4f} | Acc: {100 * correct / total:.2f}%")

    # 4. Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()