import uvicorn
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Open http://localhost:8000 in your browser.")
    # Reload=True allows editing files in src/ without restarting
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)