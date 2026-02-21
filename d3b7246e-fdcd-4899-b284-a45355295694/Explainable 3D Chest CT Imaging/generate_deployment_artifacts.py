
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "/tmp/lung_nodule_api"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dockerfile  — multi-stage, CPU-only torch
# ─────────────────────────────────────────────────────────────────────────────
DOCKERFILE = r"""# ── Stage 1: builder – install heavy deps into a virtual env ─────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create venv and install all packages (CPU-only torch index)
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip --quiet && \
    /opt/venv/bin/pip install --quiet \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="lung-nodule-api" \
      version="1.0.0" \
      description="3D CNN Renal CT Inference API (LungNoduleNet3D)"

# Minimal runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Make venv the default Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy application code
COPY main.py .

# Model weights mount point (override at runtime via -v)
RUN mkdir -p /app/weights

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)" || exit 1

# Launch uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. main.py  — complete, self-contained FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
MAIN_PY = r'''"""
main.py  —  3D CNN Renal CT Inference API
LungNoduleNet3D · ResNet-style 3D patch classifier with Grad-CAM explainability

Endpoints:
  GET  /          HTML clinical UI
  GET  /health    Model status JSON
  POST /predict   NIfTI upload → class + confidence
  POST /gradcam   NIfTI upload → Grad-CAM PNG (base64)
  GET  /docs      Swagger / OpenAPI docs
"""

import io
import os
import base64
import tempfile
import json

import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import nibabel as nib
import torch
import torch.nn as nn

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/weights/lung_nodule_3d_resnet.pth")
DEVICE      = torch.device("cpu")
HU_MIN      = -1000.0
HU_MAX      =  400.0
TGT_SPACING = (1.0, 1.0, 1.0)   # mm — isotropic resample target
PATCH_SIZE  = (64, 64, 64)
HALF        = tuple(p // 2 for p in PATCH_SIZE)
LABEL_NAMES = {0: "background", 1: "kidney / tissue", 2: "tumour"}

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    """3-D residual block: two 3×3×3 conv layers with a skip connection."""
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class LungNoduleNet3D(nn.Module):
    """
    Lightweight 3-D ResNet for renal CT patch classification.
    Input:  (B, 1, 64, 64, 64) normalised HU patch
    Output: (B, num_classes) logits
    """
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
            ResBlock3D(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64),
        )
        self.layer3 = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Load model weights at startup
# ─────────────────────────────────────────────────────────────────────────────
_srv_model    = None
_model_status = "not_loaded"

if os.path.exists(MODEL_PATH):
    _srv_model = LungNoduleNet3D(num_classes=2).to(DEVICE)
    _srv_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    _srv_model.eval()
    _model_status = "loaded"
    _n_params = sum(p.numel() for p in _srv_model.parameters())
    print(f"✓ Model loaded from {MODEL_PATH}  ({_n_params:,} parameters)")
else:
    _model_status = "weights_not_found"
    print(f"✗ Model weights not found at {MODEL_PATH}")
    print("  Mount weights with: -v /host/path/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_nifti(nii_bytes: bytes):
    """
    bytes → tmp file → nibabel → HU window → normalise → resample to 1 mm³.
    Returns (volume: ndarray, original_spacing: tuple).
    """
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(nii_bytes)
        fpath = f.name
    nii  = nib.load(fpath)
    os.unlink(fpath)
    vol  = nii.get_fdata(dtype=np.float32)
    sp   = tuple(float(np.abs(nii.header.get_zooms()[i])) for i in range(3))
    vol  = np.clip(vol, HU_MIN, HU_MAX)
    vol  = (vol - HU_MIN) / (HU_MAX - HU_MIN)        # → [0, 1]
    zf   = tuple(sp[i] / TGT_SPACING[i] for i in range(3))
    vol  = zoom(vol, zf, order=1)
    return vol, sp


def extract_centre_patch(vol: np.ndarray) -> np.ndarray:
    """Extract a single PATCH_SIZE³ patch centred on the volume (clamped)."""
    D, H, W = vol.shape
    cz = max(HALF[0], min(D // 2, D - HALF[0]))
    cy = max(HALF[1], min(H // 2, H - HALF[1]))
    cx = max(HALF[2], min(W // 2, W - HALF[2]))
    return vol[cz - HALF[0]:cz + HALF[0],
               cy - HALF[1]:cy + HALF[1],
               cx - HALF[2]:cx + HALF[2]].astype(np.float32)


def run_inference(patch: np.ndarray):
    """Run forward pass and return (class_id, class_name, confidence, probs)."""
    t = torch.from_numpy(patch[None, None]).float()
    with torch.no_grad():
        probs = torch.softmax(_srv_model(t), dim=1)[0]
    cls_id = int(probs.argmax())
    return cls_id, LABEL_NAMES.get(cls_id, "unknown"), float(probs[cls_id]), probs.tolist()


def compute_gradcam(patch: np.ndarray) -> np.ndarray:
    """
    Grad-CAM on layer3 (last residual stage) targeting class 1 (tumour/tissue).
    Returns a normalised [0, 1] heatmap with same spatial dims as `patch`.
    """
    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts["l3"] = out

    def bwd_hook(m, grad_in, grad_out):
        grads["l3"] = grad_out[0]

    target_layer = _srv_model.layer3[3]  # ResBlock3D inside layer3
    hf = target_layer.register_forward_hook(fwd_hook)
    hb = target_layer.register_full_backward_hook(bwd_hook)

    t = torch.from_numpy(patch[None, None]).float()
    _srv_model.zero_grad()
    logits = _srv_model(t)
    logits[0, 1].backward()  # target tumour/tissue class

    A   = acts["l3"]
    G   = grads["l3"]
    alp = G.mean(dim=(2, 3, 4), keepdim=True)
    raw = torch.clamp((alp * A).sum(dim=1).squeeze(), min=0)
    lo, hi = raw.min(), raw.max()
    norm = ((raw - lo) / (hi - lo + 1e-8)).detach().numpy()
    zf   = tuple(patch.shape[k] / norm.shape[k] for k in range(3))
    heat = np.clip(zoom(norm, zf, order=1), 0, 1)

    hf.remove()
    hb.remove()
    return heat


def render_gradcam_png(patch: np.ndarray, heat: np.ndarray) -> str:
    """Render a 3×3 Grad-CAM figure (CT / heatmap / overlay) and return base64 PNG."""
    BG, TXT, SEC = "#1D1D20", "#fbfbff", "#909094"
    gc_cmap = mcolors.LinearSegmentedColormap.from_list("gc_z", [
        (0.00, (0.00, 0.00, 0.00, 0.00)),
        (0.30, (0.25, 0.00, 0.50, 0.60)),
        (0.60, (1.00, 0.50, 0.00, 0.80)),
        (1.00, (1.00, 0.84, 0.00, 1.00)),
    ])
    nz     = patch.shape[0]
    slices = [nz // 4, nz // 2, 3 * nz // 4]
    peak   = np.unravel_index(np.argmax(heat), heat.shape)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.patch.set_facecolor(BG)
    fig.suptitle("3D Grad-CAM  ·  Axial Slices (z = 25 % / 50 % / 75 %)",
                 color=TXT, fontsize=14, fontweight="bold")

    for row, sz in enumerate(slices):
        ax = axes[row]
        for a in ax:
            a.set_facecolor(BG)
        ax[0].imshow(patch[sz], cmap="gray", vmin=0, vmax=1, origin="lower")
        ax[0].set_title(f"CT  z={sz}", color=TXT, fontsize=10)
        ax[0].axis("off")
        im = ax[1].imshow(heat[sz], cmap="hot", vmin=0, vmax=1, origin="lower")
        ax[1].set_title(f"Grad-CAM  z={sz}", color=TXT, fontsize=10)
        ax[1].axis("off")
        cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=SEC, labelsize=7)
        cb.outline.set_edgecolor(SEC)
        ax[2].imshow(patch[sz], cmap="gray", vmin=0, vmax=1, origin="lower")
        ax[2].imshow(heat[sz], cmap=gc_cmap, vmin=0, vmax=1, alpha=0.65, origin="lower")
        if sz == peak[0]:
            ax[2].scatter(peak[2], peak[1], c="#ffd400", s=180,
                          marker="*", edgecolors=BG, zorder=5, label="Peak")
            ax[2].legend(facecolor="#2a2a2e", edgecolor=SEC,
                         labelcolor=TXT, fontsize=8)
        ax[2].set_title(f"Overlay  z={sz}", color=TXT, fontsize=10)
        ax[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Embedded HTML frontend (single-page clinical UI)
# ─────────────────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>3D CNN · Renal CT Inference</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f12;color:#fbfbff;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
header{background:linear-gradient(135deg,#1a1a24,#12121a);border-bottom:1px solid #2a2a38;padding:1.5rem 2rem;display:flex;align-items:center;gap:1rem}
header h1{font-size:1.4rem;font-weight:700}
header .sub{font-size:.75rem;color:#909094;margin-top:.2rem}
.badge{background:#17b26a22;color:#17b26a;border:1px solid #17b26a44;border-radius:999px;font-size:.7rem;padding:.25rem .7rem;font-weight:600;margin-left:auto}
.container{max-width:1100px;margin:0 auto;padding:2rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.full{grid-column:1/-1}
.card{background:#1D1D20;border:1px solid #2a2a38;border-radius:12px;padding:1.5rem}
.card h2{font-size:.85rem;font-weight:700;margin-bottom:1rem;color:#909094;text-transform:uppercase;letter-spacing:.08em}
label{font-size:.85rem;color:#909094;display:block;margin-bottom:.4rem}
input[type=file]{width:100%;background:#12121a;border:2px dashed #2a2a38;border-radius:8px;padding:1.5rem;color:#909094;cursor:pointer;font-size:.85rem;transition:border-color .2s}
input[type=file]:hover{border-color:#A1C9F4}
.btn{display:inline-block;padding:.65rem 1.4rem;border-radius:8px;font-size:.875rem;font-weight:600;cursor:pointer;border:none;transition:opacity .2s;margin-top:.75rem}
.btn-p{background:#A1C9F4;color:#0f0f12}.btn-p:hover{opacity:.85}
.btn-g{background:#D0BBFF;color:#0f0f12}.btn-g:hover{opacity:.85}
.btn:disabled{opacity:.35;cursor:not-allowed}
.rbox{background:#12121a;border:1px solid #2a2a38;border-radius:8px;padding:1rem;margin-top:1rem;min-height:64px}
.tag{display:inline-block;padding:.25rem .75rem;border-radius:6px;font-size:.8rem;font-weight:700;margin-right:.5rem}
.t-fg{background:#17b26a22;color:#17b26a;border:1px solid #17b26a44}
.t-bg{background:#f0443822;color:#f04438;border:1px solid #f0443844}
.t-tu{background:#ffd40022;color:#ffd400;border:1px solid #ffd40044}
.row{display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid #2a2a38;font-size:.85rem}
.row:last-child{border:none}
.row .k{color:#909094}.row .v{color:#fbfbff;font-weight:600}
.sbar{background:#1D1D20;border:1px solid #2a2a38;border-radius:8px;padding:.75rem 1rem;display:flex;align-items:center;gap:.5rem;margin-bottom:1.5rem;font-size:.85rem}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:.35rem}
.dok{background:#17b26a}.derr{background:#f04438}
.spin{display:none;width:16px;height:16px;border:3px solid #2a2a38;border-top-color:#A1C9F4;border-radius:50%;animation:sp .6s linear infinite;margin-left:.5rem;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
.hmwrap{text-align:center;margin-top:1rem}
.hmwrap img{max-width:100%;border-radius:8px;border:1px solid #2a2a38}
</style>
</head>
<body>
<header>
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
    <rect width="32" height="32" rx="8" fill="#A1C9F4" fill-opacity=".12"/>
    <path d="M16 5 L27 11.5 L27 20.5 L16 27 L5 20.5 L5 11.5Z" stroke="#A1C9F4" stroke-width="1.5" fill="none"/>
    <circle cx="16" cy="16" r="4" fill="#A1C9F4"/>
  </svg>
  <div><h1>3D CNN · Renal CT Inference</h1><div class="sub">LungNoduleNet3D · ResNet-style 3D patch classifier</div></div>
  <span class="badge" id="mbadge">Loading…</span>
</header>
<div class="container">
  <div class="sbar">
    <span class="dot" id="hdot"></span>
    <span id="htext">Checking model status…</span>
  </div>
  <div class="grid">
    <div class="card">
      <h2>🧪 Predict</h2>
      <label>Upload NIfTI file (.nii or .nii.gz)</label>
      <input type="file" id="pfile" accept=".nii,.gz">
      <button class="btn btn-p" id="pbtn" onclick="runPredict()">Run Inference <span class="spin" id="pspin"></span></button>
      <div class="rbox" id="pres"><span style="color:#909094;font-size:.85rem">Results will appear here…</span></div>
    </div>
    <div class="card">
      <h2>🔥 Grad-CAM Explainability</h2>
      <label>Upload NIfTI file (.nii or .nii.gz)</label>
      <input type="file" id="gfile" accept=".nii,.gz">
      <button class="btn btn-g" id="gbtn" onclick="runGradcam()">Generate Heatmap <span class="spin" id="gspin"></span></button>
      <div class="rbox" id="gres"><span style="color:#909094;font-size:.85rem">Heatmap stats will appear here…</span></div>
    </div>
    <div class="card full" id="hmcard" style="display:none">
      <h2>📊 Grad-CAM Heatmap · 3 Axial Slices</h2>
      <div class="hmwrap"><img id="hmimg" src="" alt="Grad-CAM"/></div>
    </div>
    <div class="card full">
      <h2>📡 API Reference</h2>
      <div class="row"><span class="k">GET  /health</span><span class="v">Model status JSON</span></div>
      <div class="row"><span class="k">POST /predict</span><span class="v">NIfTI upload → class + confidence</span></div>
      <div class="row"><span class="k">POST /gradcam</span><span class="v">NIfTI upload → Grad-CAM PNG (base64)</span></div>
      <div class="row"><span class="k">GET  /docs</span><span class="v">Swagger / OpenAPI</span></div>
      <div class="row"><span class="k">Model</span><span class="v">LungNoduleNet3D · 1.47M params</span></div>
      <div class="row"><span class="k">Input</span><span class="v">64³ voxels · 1 mm isotropic</span></div>
      <div class="row"><span class="k">Classes</span><span class="v">0=background · 1=kidney/tissue · 2=tumour</span></div>
      <div class="row"><span class="k">HU window</span><span class="v">−1000 to +400 HU</span></div>
    </div>
  </div>
</div>
<script>
async function checkHealth(){
  const r=await fetch('/health'),d=await r.json(),ok=d.status==='ok';
  document.getElementById('hdot').className='dot '+(ok?'dok':'derr');
  document.getElementById('htext').textContent='Model: '+d.model_status+' · Device: '+d.device+' · Params: '+d.parameters.toLocaleString();
  document.getElementById('mbadge').textContent=ok?'● Model Ready':'✗ Model Error';
  document.getElementById('mbadge').style.color=ok?'#17b26a':'#f04438';
}
async function runPredict(){
  const f=document.getElementById('pfile').files[0];
  if(!f){alert('Please select a NIfTI file.');return;}
  const btn=document.getElementById('pbtn'),sp=document.getElementById('pspin');
  btn.disabled=true;sp.style.display='inline-block';
  document.getElementById('pres').innerHTML='<span style="color:#909094;font-size:.85rem">Running inference…</span>';
  const fd=new FormData();fd.append('file',f);
  const r=await fetch('/predict',{method:'POST',body:fd}),d=await r.json();
  btn.disabled=false;sp.style.display='none';
  if(d.detail){document.getElementById('pres').innerHTML=`<span style="color:#f04438">Error: ${d.detail}</span>`;return;}
  const tag=d.class_id===0?'t-bg':(d.class_id===2?'t-tu':'t-fg');
  const probs=(d.probabilities||[]).map((p,i)=>`<div class="row"><span class="k">Class ${i}: ${['Background','Kidney/Tissue','Tumour'][i]||'?'}</span><span class="v">${(p*100).toFixed(1)}%</span></div>`).join('');
  document.getElementById('pres').innerHTML=`<div style="margin-bottom:.75rem"><span class="tag ${tag}">${(d.class_name||'').toUpperCase()}</span><span style="font-size:.8rem;color:#909094">Confidence: <strong style="color:#fbfbff">${(d.confidence*100).toFixed(1)}%</strong></span></div>${probs}<div class="row"><span class="k">Resampled shape</span><span class="v">${(d.resampled_shape||[]).join('×')}</span></div><div class="row"><span class="k">Orig spacing (mm)</span><span class="v">${(d.original_spacing_mm||[]).map(v=>v.toFixed(2)).join(' × ')}</span></div>`;
}
async function runGradcam(){
  const f=document.getElementById('gfile').files[0];
  if(!f){alert('Please select a NIfTI file.');return;}
  const btn=document.getElementById('gbtn'),sp=document.getElementById('gspin');
  btn.disabled=true;sp.style.display='inline-block';
  document.getElementById('gres').innerHTML='<span style="color:#909094;font-size:.85rem">Computing Grad-CAM… (~10 s)</span>';
  const fd=new FormData();fd.append('file',f);
  const r=await fetch('/gradcam',{method:'POST',body:fd}),d=await r.json();
  btn.disabled=false;sp.style.display='none';
  if(d.detail){document.getElementById('gres').innerHTML=`<span style="color:#f04438">Error: ${d.detail}</span>`;return;}
  document.getElementById('gres').innerHTML=`<div class="row"><span class="k">Peak voxel (z,y,x)</span><span class="v">(${d.peak_z}, ${d.peak_y}, ${d.peak_x})</span></div><div class="row"><span class="k">Peak activation</span><span class="v">${(d.peak_value*100).toFixed(1)}%</span></div>`;
  document.getElementById('hmimg').src='data:image/png;base64,'+d.heatmap_b64;
  document.getElementById('hmcard').style.display='block';
}
checkHealth();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="3D CNN Renal CT Inference",
    description="LungNoduleNet3D — kidney/tumour patch classifier with Grad-CAM",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """Serve the embedded clinical UI."""
    return HTMLResponse(content=_HTML)


@app.get("/health", summary="Health & model status")
async def health():
    """Return model load status, device, parameter count, and configuration."""
    loaded  = _srv_model is not None
    n_params = int(sum(p.numel() for p in _srv_model.parameters())) if loaded else 0
    return JSONResponse({
        "status":        "ok" if loaded else "error",
        "model_status":  _model_status,
        "model_path":    MODEL_PATH,
        "device":        str(DEVICE),
        "parameters":    n_params,
        "patch_size":    list(PATCH_SIZE),
        "classes":       LABEL_NAMES,
        "hu_window":     [HU_MIN, HU_MAX],
    })


@app.post("/predict", summary="Run 3D CNN inference on a NIfTI volume")
async def predict(file: UploadFile = File(..., description="NIfTI file (.nii or .nii.gz)")):
    """
    Upload a NIfTI CT volume and receive class prediction + probability scores.

    Processing pipeline:
    1. HU window clamp [−1000, +400]
    2. Normalise to [0, 1]
    3. Resample to 1 mm³ isotropic
    4. Extract 64³ centre patch
    5. Forward pass through LungNoduleNet3D
    """
    if _srv_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")
    raw_bytes     = await file.read()
    vol, spacing  = preprocess_nifti(raw_bytes)
    patch         = extract_centre_patch(vol)
    cls_id, cls_name, conf, probs = run_inference(patch)
    return JSONResponse({
        "class_id":            cls_id,
        "class_name":          cls_name,
        "confidence":          round(conf, 6),
        "probabilities":       [round(float(p), 6) for p in probs],
        "original_spacing_mm": list(spacing),
        "resampled_shape":     list(vol.shape),
        "patch_centre":        [d // 2 for d in vol.shape],
    })


@app.post("/gradcam", summary="Generate Grad-CAM explainability heatmap")
async def gradcam(file: UploadFile = File(..., description="NIfTI file (.nii or .nii.gz)")):
    """
    Upload a NIfTI CT volume and receive a Grad-CAM activation heatmap.

    Returns base64-encoded PNG showing 3 axial slices (25 % / 50 % / 75 %)
    with CT, raw heatmap, and overlay columns.  Peak activation coordinates
    are also returned.
    """
    if _srv_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")
    raw_bytes    = await file.read()
    vol, spacing = preprocess_nifti(raw_bytes)
    patch        = extract_centre_patch(vol)
    heat         = compute_gradcam(patch)
    b64          = render_gradcam_png(patch, heat)
    peak         = np.unravel_index(np.argmax(heat), heat.shape)
    return JSONResponse({
        "heatmap_b64":   b64,
        "peak_z":        int(peak[0]),
        "peak_y":        int(peak[1]),
        "peak_x":        int(peak[2]),
        "peak_value":    float(heat[peak]),
        "heatmap_shape": list(heat.shape),
    })
'''

# ─────────────────────────────────────────────────────────────────────────────
# 3. requirements.txt  — all pinned dependencies
# ─────────────────────────────────────────────────────────────────────────────
REQUIREMENTS_TXT = """\
# ── Core framework ─────────────────────────────────────────────────────────
fastapi==0.111.1
uvicorn[standard]==0.30.3
python-multipart==0.0.9

# ── Deep learning (CPU-only; use --extra-index-url for torch) ──────────────
# Install via:
#   pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
torch==2.3.1+cpu
torchvision==0.18.1+cpu

# ── Medical imaging ────────────────────────────────────────────────────────
nibabel==5.2.1

# ── Numerics & image processing ────────────────────────────────────────────
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.0
scikit-learn==1.5.0

# ── HTTP server extras (bundled in uvicorn[standard]) ─────────────────────
httptools==0.6.1
uvloop==0.19.0; sys_platform != "win32"
websockets==12.0
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4. docker-compose.yml
# ─────────────────────────────────────────────────────────────────────────────
DOCKER_COMPOSE = """\
version: "3.9"

services:
  lung-nodule-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: lung-nodule-api:latest
    container_name: lung_nodule_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      # Mount your trained model weights here
      - ./weights/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro
    environment:
      - MODEL_PATH=/app/weights/lung_nodule_3d_resnet.pth
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)"]
      interval: 30s
      timeout: 10s
      start_period: 20s
      retries: 3
    mem_limit: 4g
    # For GPU support, uncomment the following (requires nvidia-container-toolkit):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. .dockerignore
# ─────────────────────────────────────────────────────────────────────────────
DOCKERIGNORE = """\
# Version control
.git
.gitignore
.github

# Python artefacts
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
env/
.env

# Jupyter / IDEs
.ipynb_checkpoints/
*.ipynb
.idea/
.vscode/
*.swp
*.swo

# Large data / model files (mount at runtime instead)
*.nii
*.nii.gz
*.pth
*.pt
*.ckpt
*.h5
weights/
data/

# Logs & temp
*.log
logs/
tmp/
.tmp/

# OS files
.DS_Store
Thumbs.db

# Documentation source (already in README.md)
docs/
*.rst
"""

# ─────────────────────────────────────────────────────────────────────────────
# 6. deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
DEPLOY_SH = """\
#!/usr/bin/env bash
# deploy.sh — Build and run the Lung Nodule API Docker container
# Usage: ./deploy.sh [--weights /path/to/lung_nodule_3d_resnet.pth]
set -euo pipefail

IMAGE_NAME="lung-nodule-api"
IMAGE_TAG="latest"
CONTAINER_NAME="lung_nodule_api"
PORT=8000
WEIGHTS_PATH="./weights/lung_nodule_3d_resnet.pth"

# ── Parse optional --weights flag ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║   3D CNN Renal CT Inference API — Deploy Script   ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── 1. Build Docker image ─────────────────────────────────────────────────────
echo "▶ Building Docker image ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build \
  --target runtime \
  --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
  --file Dockerfile \
  .
echo "✓ Image built successfully"
echo ""

# ── 2. Stop & remove existing container (if any) ──────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "▶ Stopping existing container '${CONTAINER_NAME}' ..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm   "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "✓ Old container removed"
  echo ""
fi

# ── 3. Prepare weights volume ─────────────────────────────────────────────────
WEIGHTS_MOUNT_FLAG=""
if [[ -f "${WEIGHTS_PATH}" ]]; then
  ABS_WEIGHTS=$(realpath "${WEIGHTS_PATH}")
  WEIGHTS_MOUNT_FLAG="-v ${ABS_WEIGHTS}:/app/weights/lung_nodule_3d_resnet.pth:ro"
  echo "✓ Model weights found: ${ABS_WEIGHTS}"
else
  echo "⚠ Warning: weights file not found at '${WEIGHTS_PATH}'"
  echo "  The server will start but /predict and /gradcam will return 503."
  echo "  Mount weights with: --weights /path/to/lung_nodule_3d_resnet.pth"
fi
echo ""

# ── 4. Run container ──────────────────────────────────────────────────────────
echo "▶ Starting container '${CONTAINER_NAME}' on port ${PORT} ..."
docker run \
  --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --publish "${PORT}:8000" \
  --env MODEL_PATH=/app/weights/lung_nodule_3d_resnet.pth \
  --env PYTHONUNBUFFERED=1 \
  --memory 4g \
  ${WEIGHTS_MOUNT_FLAG} \
  "${IMAGE_NAME}:${IMAGE_TAG}"

echo "✓ Container started (id: $(docker ps -q -f name=${CONTAINER_NAME}))"
echo ""

# ── 5. Wait for health check ──────────────────────────────────────────────────
echo "▶ Waiting for server to become healthy ..."
MAX_WAIT=60
ELAPSED=0
until docker exec "${CONTAINER_NAME}" python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" \
    >/dev/null 2>&1; do
  sleep 3
  ELAPSED=$((ELAPSED + 3))
  if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
    echo "✗ Server did not become healthy within ${MAX_WAIT}s"
    echo "  Check logs: docker logs ${CONTAINER_NAME}"
    exit 1
  fi
  echo "  ... still waiting (${ELAPSED}s)"
done

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║   ✓  Server is RUNNING                            ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "  🌐 Clinical UI  →  http://localhost:${PORT}"
echo "  📡 API docs     →  http://localhost:${PORT}/docs"
echo "  ❤  Health       →  http://localhost:${PORT}/health"
echo ""
echo "  Useful commands:"
echo "    docker logs -f ${CONTAINER_NAME}      # live logs"
echo "    docker stop ${CONTAINER_NAME}         # stop server"
echo "    docker stats ${CONTAINER_NAME}        # resource usage"
echo ""
"""

# ─────────────────────────────────────────────────────────────────────────────
# 7. README.md
# ─────────────────────────────────────────────────────────────────────────────
README_MD = """\
# 3D CNN Renal CT Inference API

> **LungNoduleNet3D** — a lightweight 3-D ResNet patch classifier for kidney/tumour detection in renal CT scans, served via FastAPI with Grad-CAM explainability.

---

## 📋 Contents

```
lung_nodule_api/
├── Dockerfile            # Multi-stage Docker build (CPU-only torch)
├── main.py               # Complete FastAPI application
├── requirements.txt      # Pinned Python dependencies
├── docker-compose.yml    # Single-service Compose definition
├── .dockerignore         # Files excluded from Docker build context
├── deploy.sh             # One-click build + run script
└── README.md             # This file
```

---

## 🚀 Quick Start

### Option A — deploy.sh (recommended)

```bash
# 1. Place your trained model weights
mkdir -p weights
cp /path/to/lung_nodule_3d_resnet.pth weights/

# 2. Make deploy script executable and run it
chmod +x deploy.sh
./deploy.sh --weights ./weights/lung_nodule_3d_resnet.pth
```

The script will build the image, start the container, wait for the health check to pass, and print the access URLs.

---

### Option B — Docker CLI

```bash
# Build
docker build --target runtime -t lung-nodule-api:latest .

# Run (with weights mounted)
docker run -d \\
  --name lung_nodule_api \\
  --restart unless-stopped \\
  -p 8000:8000 \\
  -v $(pwd)/weights/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro \\
  -e MODEL_PATH=/app/weights/lung_nodule_3d_resnet.pth \\
  --memory 4g \\
  lung-nodule-api:latest
```

---

### Option C — Docker Compose

```bash
mkdir -p weights
cp /path/to/lung_nodule_3d_resnet.pth weights/
docker compose up -d
```

---

## 🌐 Accessing the Server

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Interactive clinical UI |
| `http://localhost:8000/health` | Model status JSON |
| `http://localhost:8000/docs` | Swagger / OpenAPI interactive docs |
| `http://localhost:8000/redoc` | ReDoc API documentation |

---

## 📡 API Endpoints

### `GET /health`

Returns model load status, device, and configuration.

```json
{
  "status": "ok",
  "model_status": "loaded",
  "model_path": "/app/weights/lung_nodule_3d_resnet.pth",
  "device": "cpu",
  "parameters": 1466770,
  "patch_size": [64, 64, 64],
  "classes": {"0": "background", "1": "kidney / tissue", "2": "tumour"},
  "hu_window": [-1000.0, 400.0]
}
```

---

### `POST /predict`

Upload a NIfTI CT file to receive classification results.

```bash
curl -X POST http://localhost:8000/predict \\
  -F "file=@patient001.nii.gz"
```

**Response:**

```json
{
  "class_id": 2,
  "class_name": "tumour",
  "confidence": 0.923456,
  "probabilities": [0.012, 0.065, 0.923],
  "original_spacing_mm": [0.625, 0.625, 1.5],
  "resampled_shape": [420, 512, 512],
  "patch_centre": [210, 256, 256]
}
```

---

### `POST /gradcam`

Upload a NIfTI CT file to receive a Grad-CAM explainability heatmap.

```bash
curl -X POST http://localhost:8000/gradcam \\
  -F "file=@patient001.nii.gz" \\
  -o gradcam_response.json
```

**Response:**

```json
{
  "heatmap_b64": "<base64-encoded PNG>",
  "peak_z": 32,
  "peak_y": 28,
  "peak_x": 31,
  "peak_value": 1.0,
  "heatmap_shape": [64, 64, 64]
}
```

Decode and view the PNG:

```python
import base64, json
data = json.load(open('gradcam_response.json'))
with open('heatmap.png', 'wb') as f:
    f.write(base64.b64decode(data['heatmap_b64']))
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/weights/lung_nodule_3d_resnet.pth` | Path to trained `.pth` weights inside the container |
| `PYTHONUNBUFFERED` | `1` | Force stdout flush for live logs |
| `PYTHONDONTWRITEBYTECODE` | `1` | Avoid `.pyc` files |

---

## 🏋️ Mounting Model Weights

The model weights file must be mounted into the container at startup. The default expected path is `/app/weights/lung_nodule_3d_resnet.pth`.

**Docker CLI:**

```bash
-v /host/path/to/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro
```

**Docker Compose** (`docker-compose.yml` already includes):

```yaml
volumes:
  - ./weights/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro
```

---

## 🩺 Model Details

| Property | Value |
|----------|-------|
| Architecture | LungNoduleNet3D (3-D ResNet) |
| Parameters | ~1.47M |
| Input | 64×64×64 voxel patch, 1-channel |
| Pre-processing | HU clamp [−1000, +400], normalise [0,1], resample to 1 mm³ |
| Classes | 0=background, 1=kidney/tissue, 2=tumour |
| Device | CPU (GPU-ready, see docker-compose.yml) |
| Framework | PyTorch 2.x |

---

## 🛑 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` returns `model_status: weights_not_found` | Check the weights path and volume mount |
| `POST /predict` returns HTTP 503 | Model not loaded — see above |
| Container OOM killed | Increase `--memory` limit (default: 4 g) |
| Port 8000 already in use | Change host port: `-p 8080:8000` |
| Slow inference (>30 s) | Normal for CPU; consider GPU deployment |

---

## 📜 Licence

Research / educational use only. Not for clinical diagnosis.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Write all files to disk
# ─────────────────────────────────────────────────────────────────────────────
artifact_files = {
    "Dockerfile":          DOCKERFILE,
    "main.py":             MAIN_PY,
    "requirements.txt":    REQUIREMENTS_TXT,
    "docker-compose.yml":  DOCKER_COMPOSE,
    ".dockerignore":       DOCKERIGNORE,
    "deploy.sh":           DEPLOY_SH,
    "README.md":           README_MD,
}

print("=" * 70)
print(f"  Writing deployment artifacts to {OUT_DIR}")
print("=" * 70)

for filename, content in artifact_files.items():
    fpath = os.path.join(OUT_DIR, filename)
    with open(fpath, "w", encoding="utf-8") as fh:
        fh.write(content)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  ✓  {filename:<25}  ({size_kb:.1f} KB)")

# Make deploy.sh executable
os.chmod(os.path.join(OUT_DIR, "deploy.sh"), 0o755)
print(f"\n  ✓  deploy.sh marked executable (chmod +x)")

# ─────────────────────────────────────────────────────────────────────────────
# Verify directory structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Directory listing: os.listdir('/tmp/lung_nodule_api/')")
print("=" * 70)
deploy_files_list = sorted(os.listdir(OUT_DIR))
for fname in deploy_files_list:
    fpath = os.path.join(OUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:<30}  {size_kb:6.1f} KB")
print(f"\n  Total: {len(deploy_files_list)} files")

# ─────────────────────────────────────────────────────────────────────────────
# Print each file's content
# ─────────────────────────────────────────────────────────────────────────────
SEP = "─" * 70

for filename in ["Dockerfile", "main.py", "requirements.txt",
                 "docker-compose.yml", ".dockerignore", "deploy.sh", "README.md"]:
    fpath = os.path.join(OUT_DIR, filename)
    with open(fpath, "r", encoding="utf-8") as fh:
        file_content = fh.read()
    print(f"\n\n{'=' * 70}")
    print(f"  FILE: {filename}")
    print(f"{'=' * 70}")
    print(file_content)
    print(SEP)

print("\n✅  All 7 deployment artifacts written and printed successfully.")
print(f"   Location: {OUT_DIR}")
