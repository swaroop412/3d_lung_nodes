import os
# The following line is a workaround for a common issue on Windows with Anaconda,
# where multiple OpenMP runtimes can be loaded, causing a crash.
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from model import LungNoduleNet3D
from config import MODEL_PATH, LABELS, DEVICE, NUM_CLASSES
from utils import preprocess_nifti, centre_patch, compute_gradcam, render_gradcam_animation

app = FastAPI(title="3D CNN Lung Nodule Detection")

model = None
model_status = "not_loaded"

@app.on_event("startup")
def load_model():
    """Load the PyTorch model at application startup."""
    global model, model_status
    try:
        model = LungNoduleNet3D(num_classes=NUM_CLASSES).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            # Load trained weights if they exist
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model_status = "loaded_from_disk"
        else:
            # If no weights, the model is initialized but untrained
            model_status = "initialized_no_weights"
        model.eval()
    except Exception as e:
        model_status = f"error_loading: {str(e)}"
        model = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML frontend."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
async def health_check():
    """Check the status of the model."""
    params = sum(p.numel() for p in model.parameters()) if model else 0
    return JSONResponse({
        "model_status": model_status,
        "device": str(DEVICE),
        "parameters": f"{params/1e6:.2f}M",
        "labels": LABELS
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for nodule detection."""
    if not model:
        raise HTTPException(503, "Model is not available.")
    
    contents = await file.read()
    volume, spacing = preprocess_nifti(contents)
    patch = centre_patch(volume)
    
    tensor = torch.from_numpy(patch[None, None]).float().to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    pred_idx = probs.argmax().item()
    return JSONResponse({
        "class_id": pred_idx,
        "class_name": LABELS.get(pred_idx, "Unknown"),
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {LABELS[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    })

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """Endpoint for generating Grad-CAM explainability visualization."""
    if not model:
        raise HTTPException(503, "Model is not available.")
        
    contents = await file.read()
    volume, _ = preprocess_nifti(contents)
    patch = centre_patch(volume)
    heatmap = compute_gradcam(model, patch)
    animations = render_gradcam_animation(patch, heatmap)
    
    return JSONResponse(animations)