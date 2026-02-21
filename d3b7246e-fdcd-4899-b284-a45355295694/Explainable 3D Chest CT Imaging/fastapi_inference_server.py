
import subprocess, sys, os, threading, time

# ── MUST insert path BEFORE any third-party imports ─────────────────────────
if "/tmp/pypackages" not in sys.path:
    sys.path.insert(0, "/tmp/pypackages")

# ── Install all deps (torch first, since it's largest) ───────────────────────
def _pip(*args):
    return subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--target=/tmp/pypackages", "--quiet", *args],
        capture_output=True, text=True
    )

_r_torch = _pip("torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu")
print("torch install  →", _r_torch.returncode)

for _p in ["fastapi", "uvicorn[standard]", "python-multipart", "nibabel"]:
    _r = _pip(_p)
    print(f"{_p:<25} → {_r.returncode}")

# ── Core imports ─────────────────────────────────────────────────────────────
import io, base64, tempfile, json
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import nibabel as nib
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

print(f"\nFastAPI {fastapi.__version__}  |  PyTorch {torch.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL ARCHITECTURE  (must match training exactly)
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
        x = self.stem(x);   x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x);   return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
_SERVER_MODEL_PATH = "/tmp/lung_nodule_3d_resnet.pth"
_srv_device   = torch.device("cpu")
_srv_model    = None
_model_status = "not_loaded"

if os.path.exists(_SERVER_MODEL_PATH):
    _srv_model = _LungNoduleNet3D(num_classes=2).to(_srv_device)
    _srv_model.load_state_dict(
        torch.load(_SERVER_MODEL_PATH, map_location=_srv_device,
                   weights_only=True)
    )
    _srv_model.eval()
    _model_status = "loaded"
    print(f"✓  Model loaded  ({os.path.getsize(_SERVER_MODEL_PATH)/1024:.1f} KB)")
else:
    _model_status = "weights_not_found"
    print(f"✗  Model weights not found at {_SERVER_MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_HU_MIN = -1000.0
_HU_MAX =  400.0
_TGT_SP = (1.0, 1.0, 1.0)
_PSIZ   = (64, 64, 64)
_HALF   = tuple(p // 2 for p in _PSIZ)
_LABELS = {0: "background", 1: "kidney / tissue", 2: "tumour"}


def _preprocess_nifti(nii_bytes: bytes):
    """Save bytes → temp file → nib.load → HU window → normalise → resample."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as _f:
        _f.write(nii_bytes); _fpath = _f.name
    _nii = nib.load(_fpath); os.unlink(_fpath)
    _vol = _nii.get_fdata(dtype=np.float32)
    _sp  = tuple(float(np.abs(_nii.header.get_zooms()[i])) for i in range(3))
    _vol = np.clip(_vol, _HU_MIN, _HU_MAX)
    _vol = (_vol - _HU_MIN) / (_HU_MAX - _HU_MIN)
    _zf  = tuple(_sp[i] / _TGT_SP[i] for i in range(3))
    _vol = zoom(_vol, _zf, order=1)
    return _vol, _sp


def _centre_patch(vol: np.ndarray):
    """Extract a single 64³ patch from the volume centre (clamped)."""
    D, H, W = vol.shape
    cz = max(_HALF[0], min(D // 2, D - _HALF[0]))
    cy = max(_HALF[1], min(H // 2, H - _HALF[1]))
    cx = max(_HALF[2], min(W // 2, W - _HALF[2]))
    return vol[cz-_HALF[0]:cz+_HALF[0],
               cy-_HALF[1]:cy+_HALF[1],
               cx-_HALF[2]:cx+_HALF[2]].astype(np.float32)


def _run_inference(patch: np.ndarray):
    _t = torch.from_numpy(patch[None, None]).float()
    with torch.no_grad():
        _probs = torch.softmax(_srv_model(_t), dim=1)[0]
    _cls  = int(_probs.argmax())
    return _cls, _LABELS.get(_cls, "unknown"), float(_probs[_cls]), _probs.tolist()


def _compute_gradcam(patch: np.ndarray):
    _acts, _grads = {}, {}
    def _fwd(m, i, o): _acts["l3"] = o
    def _bwd(m, gi, go): _grads["l3"] = go[0]
    _tgt = _srv_model.layer3[3]
    _hf  = _tgt.register_forward_hook(_fwd)
    _hb  = _tgt.register_full_backward_hook(_bwd)

    _t = torch.from_numpy(patch[None, None]).float()
    _srv_model.zero_grad()
    _logits = _srv_model(_t)
    _logits[0, 1].backward()

    _A    = _acts["l3"]
    _G    = _grads["l3"]
    _alp  = _G.mean(dim=(2, 3, 4), keepdim=True)
    _raw  = torch.clamp((_alp * _A).sum(dim=1).squeeze(), min=0)
    _lo, _hi = _raw.min(), _raw.max()
    _norm = ((_raw - _lo) / (_hi - _lo + 1e-8)).detach().numpy()
    _zf   = tuple(patch.shape[k] / _norm.shape[k] for k in range(3))
    _heat = np.clip(zoom(_norm, _zf, order=1), 0, 1)

    _hf.remove(); _hb.remove()
    return _heat


def _render_gradcam_png(patch: np.ndarray, heat: np.ndarray) -> str:
    _BG, _TXT, _SEC = "#1D1D20", "#fbfbff", "#909094"
    _gc_cmap = mcolors.LinearSegmentedColormap.from_list("gc_z", [
        (0.00, (0.00, 0.00, 0.00, 0.00)),
        (0.30, (0.25, 0.00, 0.50, 0.60)),
        (0.60, (1.00, 0.50, 0.00, 0.80)),
        (1.00, (1.00, 0.84, 0.00, 1.00)),
    ])
    _nz     = patch.shape[0]
    _slices = [_nz // 4, _nz // 2, 3 * _nz // 4]
    _peak   = np.unravel_index(np.argmax(heat), heat.shape)

    _fig, _axes = plt.subplots(3, 3, figsize=(15, 15))
    _fig.patch.set_facecolor(_BG)
    _fig.suptitle("3D Grad-CAM  ·  Axial Slices (z = 25 % / 50 % / 75 %)",
                  color=_TXT, fontsize=14, fontweight="bold")

    for _row, _sz in enumerate(_slices):
        _ax = _axes[_row]
        for _a in _ax: _a.set_facecolor(_BG)
        _ax[0].imshow(patch[_sz], cmap="gray", vmin=0, vmax=1, origin="lower")
        _ax[0].set_title(f"CT  z={_sz}", color=_TXT, fontsize=10); _ax[0].axis("off")
        _im = _ax[1].imshow(heat[_sz], cmap="hot", vmin=0, vmax=1, origin="lower")
        _ax[1].set_title(f"Grad-CAM  z={_sz}", color=_TXT, fontsize=10); _ax[1].axis("off")
        _cb = _fig.colorbar(_im, ax=_ax[1], fraction=0.046, pad=0.04)
        _cb.ax.tick_params(colors=_SEC, labelsize=7); _cb.outline.set_edgecolor(_SEC)
        _ax[2].imshow(patch[_sz], cmap="gray", vmin=0, vmax=1, origin="lower")
        _ax[2].imshow(heat[_sz], cmap=_gc_cmap, vmin=0, vmax=1, alpha=0.65, origin="lower")
        if _sz == _peak[0]:
            _ax[2].scatter(_peak[2], _peak[1], c="#ffd400", s=180,
                           marker="*", edgecolors=_BG, zorder=5, label="Peak")
            _ax[2].legend(facecolor="#2a2a2e", edgecolor=_SEC, labelcolor=_TXT, fontsize=8)
        _ax[2].set_title(f"Overlay  z={_sz}", color=_TXT, fontsize=10); _ax[2].axis("off")

    plt.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight", facecolor=_BG)
    plt.close(_fig); _buf.seek(0)
    return base64.b64encode(_buf.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 4. EMBEDDED HTML FRONTEND
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
# 5. FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
_app = FastAPI(
    title="3D CNN Renal CT Inference",
    description="LungNoduleNet3D — kidney/tumour patch classifier with Grad-CAM",
    version="1.0.0",
)

@_app.get("/", response_class=HTMLResponse)
async def srv_index():
    return HTMLResponse(content=_HTML)

@_app.get("/health")
async def srv_health():
    _loaded = _srv_model is not None
    _params = int(sum(p.numel() for p in _srv_model.parameters())) if _loaded else 0
    return JSONResponse({
        "status":       "ok" if _loaded else "error",
        "model_status": _model_status,
        "model_path":   _SERVER_MODEL_PATH,
        "device":       str(_srv_device),
        "parameters":   _params,
        "patch_size":   list(_PSIZ),
        "classes":      _LABELS,
        "hu_window":    [_HU_MIN, _HU_MAX],
    })

@_app.post("/predict")
async def srv_predict(file: UploadFile = File(...)):
    if _srv_model is None:
        raise HTTPException(503, detail="Model not loaded")
    _data = await file.read()
    _vol, _sp = _preprocess_nifti(_data)
    _patch    = _centre_patch(_vol)
    _cls, _name, _conf, _probs = _run_inference(_patch)
    return JSONResponse({
        "class_id":           _cls,
        "class_name":         _name,
        "confidence":         round(_conf, 6),
        "probabilities":      [round(float(p), 6) for p in _probs],
        "original_spacing_mm": list(_sp),
        "resampled_shape":    list(_vol.shape),
        "patch_centre":       [d // 2 for d in _vol.shape],
    })

@_app.post("/gradcam")
async def srv_gradcam(file: UploadFile = File(...)):
    if _srv_model is None:
        raise HTTPException(503, detail="Model not loaded")
    _data  = await file.read()
    _vol, _sp = _preprocess_nifti(_data)
    _patch = _centre_patch(_vol)
    _heat  = _compute_gradcam(_patch)
    _b64   = _render_gradcam_png(_patch, _heat)
    _peak  = np.unravel_index(np.argmax(_heat), _heat.shape)
    return JSONResponse({
        "heatmap_b64":   _b64,
        "peak_z":        int(_peak[0]),
        "peak_y":        int(_peak[1]),
        "peak_x":        int(_peak[2]),
        "peak_value":    float(_heat[_peak]),
        "heatmap_shape": list(_heat.shape),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 6. LAUNCH SERVER IN BACKGROUND THREAD
# ─────────────────────────────────────────────────────────────────────────────
_PORT = 8000

def _run_server():
    uvicorn.run(_app, host="0.0.0.0", port=_PORT, log_level="warning")

_srv_thread = threading.Thread(target=_run_server, daemon=True)
_srv_thread.start()
time.sleep(3)   # let uvicorn bind

# ─────────────────────────────────────────────────────────────────────────────
# 7. PRINT STATUS & SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────
import socket as _sock
_hostname = _sock.gethostname()

print("\n" + "═" * 65)
print("  🚀  FastAPI Inference Server  ·  RUNNING")
print("═" * 65)
print(f"  Local URL    :  http://localhost:{_PORT}")
print(f"  Bind address :  http://0.0.0.0:{_PORT}")
print(f"  Hostname     :  {_hostname}")
print("─" * 65)
print(f"  GET  /          →  HTML clinical UI")
print(f"  GET  /health    →  Model status JSON")
print(f"  POST /predict   →  NIfTI → class + confidence")
print(f"  POST /gradcam   →  NIfTI → Grad-CAM PNG (base64)")
print(f"  GET  /docs      →  Swagger / OpenAPI docs")
print("─" * 65)
print(f"  Model status : {_model_status}")
if _srv_model:
    print(f"  Parameters   : {sum(p.numel() for p in _srv_model.parameters()):,}")
print(f"  Device       : {_srv_device}")
print(f"  Patch size   : {_PSIZ}  HU window: [{int(_HU_MIN)}, {int(_HU_MAX)}]")
print("═" * 65)

# Smoke-test /health
import urllib.request as _ur
try:
    _resp = _ur.urlopen(f"http://localhost:{_PORT}/health", timeout=8)
    _body = json.loads(_resp.read().decode())
    print(f"\n  ✓  /health smoke-test PASSED")
    print(f"     status={_body['status']}  "
          f"model={_body['model_status']}  "
          f"params={_body['parameters']:,}")
except Exception as _e:
    print(f"\n  ✗  /health smoke-test FAILED: {_e}")

print("\n  Server is running in a background thread.")
print("  It will stay alive for the duration of this runtime session.")
