import os
import zipfile

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = "/tmp/lung_nodule_api"
ZIP_PATH = "/tmp/lung_nodule_api_deployment.zip"

# ── Re-write artifact files from inherited variables ──────────────────────────
# The string variables are passed from generate_deployment_artifacts upstream
os.makedirs(SRC_DIR, exist_ok=True)

artifact_files_pkg = {
    "Dockerfile":         DOCKERFILE,
    "main.py":            MAIN_PY,
    "requirements.txt":   REQUIREMENTS_TXT,
    "docker-compose.yml": DOCKER_COMPOSE,
    ".dockerignore":      DOCKERIGNORE,
    "deploy.sh":          DEPLOY_SH,
    "README.md":          README_MD,
}

for _fn, _fc in artifact_files_pkg.items():
    _fp = os.path.join(SRC_DIR, _fn)
    with open(_fp, "w", encoding="utf-8") as _fh:
        _fh.write(_fc)

# Make deploy.sh executable
os.chmod(os.path.join(SRC_DIR, "deploy.sh"), 0o755)

# ── Create zip archive with correct relative paths ────────────────────────────
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for _fname in sorted(os.listdir(SRC_DIR)):
        _fpath   = os.path.join(SRC_DIR, _fname)
        _arcname = os.path.join("lung_nodule_api", _fname)
        zf.write(_fpath, _arcname)

# ── Archive size ──────────────────────────────────────────────────────────────
zip_size_bytes = os.path.getsize(ZIP_PATH)
zip_size_kb    = zip_size_bytes / 1024

print("=" * 65)
print(f"  📦  Archive created:  {ZIP_PATH}")
print(f"  📏  Archive size:     {zip_size_kb:.2f} KB  ({zip_size_bytes:,} bytes)")
print("=" * 65)

# ── List archive contents ─────────────────────────────────────────────────────
print("\n  Contents of lung_nodule_api_deployment.zip")
print("  " + "─" * 60)
print(f"  {'File':<45}  {'Uncompressed':>12}  {'Compressed':>10}")
print("  " + "─" * 60)

total_uncompressed = 0
total_compressed   = 0
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for info in zf.infolist():
        uncooked_kb = info.file_size / 1024
        compress_kb = info.compress_size / 1024
        total_uncompressed += info.file_size
        total_compressed   += info.compress_size
        print(f"  {info.filename:<45}  {uncooked_kb:>10.1f} KB  {compress_kb:>8.1f} KB")

print("  " + "─" * 60)
ratio = (1 - total_compressed / total_uncompressed) * 100 if total_uncompressed else 0
print(f"  {'TOTAL':<45}  {total_uncompressed/1024:>10.1f} KB  {total_compressed/1024:>8.1f} KB")
print(f"  Compression ratio: {ratio:.1f}% saved")

# ── Deployment Summary ────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  🚀  DEPLOYMENT SUMMARY — LungNoduleNet3D  3D CT Inference API")
print("=" * 65)

print("""
  IMAGE NAME
  ──────────
  lung-nodule-api:latest

  EXPOSED PORT
  ────────────
  8000 (TCP)  →  FastAPI / uvicorn

  API ENDPOINTS
  ─────────────
  GET  /           Interactive clinical HTML UI
  GET  /health     Model status JSON (load state, params, config)
  POST /predict    NIfTI upload  →  class ID + confidence + probabilities
  POST /gradcam    NIfTI upload  →  Grad-CAM heatmap PNG (base64)
  GET  /docs       Swagger / OpenAPI interactive docs
  GET  /redoc      ReDoc API documentation

  MODEL DETAILS
  ─────────────
  Architecture : LungNoduleNet3D  (3-D ResNet, ~1.47 M params)
  Input        : 64×64×64 voxel patch, 1-channel float32
  Pre-processing: HU clamp [−1000, +400] → normalise [0,1] → resample 1 mm³
  Classes      : 0=background  |  1=kidney/tissue  |  2=tumour

  MOUNTING MODEL WEIGHTS
  ──────────────────────
  Default path inside container: /app/weights/lung_nodule_3d_resnet.pth

  Option A  (Docker CLI volume flag):
    -v /host/path/to/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro

  Option B  (Docker Compose — already configured in docker-compose.yml):
    volumes:
      - ./weights/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro

  Option C  (Environment variable override):
    -e MODEL_PATH=/custom/path/model.pth

  DOCKER RUN COMMAND
  ──────────────────
  docker run -d \\
    --name lung_nodule_api \\
    --restart unless-stopped \\
    -p 8000:8000 \\
    -v $(pwd)/weights/lung_nodule_3d_resnet.pth:/app/weights/lung_nodule_3d_resnet.pth:ro \\
    -e MODEL_PATH=/app/weights/lung_nodule_3d_resnet.pth \\
    --memory 4g \\
    lung-nodule-api:latest

  QUICK START (recommended)
  ─────────────────────────
  # 1. Extract the archive
  unzip lung_nodule_api_deployment.zip

  # 2. Place model weights
  mkdir -p lung_nodule_api/weights
  cp /path/to/lung_nodule_3d_resnet.pth lung_nodule_api/weights/

  # 3. Build & deploy
  cd lung_nodule_api
  chmod +x deploy.sh
  ./deploy.sh --weights ./weights/lung_nodule_3d_resnet.pth

  # 4. Access
  #    Clinical UI  →  http://localhost:8000
  #    Health       →  http://localhost:8000/health
  #    Swagger docs →  http://localhost:8000/docs
""")
print(f"  ARCHIVE LOCATION")
print(f"  ────────────────")
print(f"  {ZIP_PATH}  ({zip_size_kb:.2f} KB)")
print("\n" + "=" * 65)
print("  ✅  Deployment packaging complete — archive ready for distribution")
print("=" * 65)
