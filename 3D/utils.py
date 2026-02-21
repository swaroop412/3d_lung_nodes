import numpy as np
import torch
import io
import os
import tempfile
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
import base64
from config import PATCH_SIZE, HU_MIN, HU_MAX

def preprocess_nifti(file_bytes: bytes):
    """
    Loads a NIfTI file from bytes, resamples/normalizes it, and returns the volume.
    """
    # Create a temporary file because nibabel usually reads from disk
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        # Get data as float32
        volume = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()
        
        # 1. Clip intensities (HU Windowing)
        volume = np.clip(volume, HU_MIN, HU_MAX)
        
        # 2. Normalize to [0, 1]
        volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
        
        return volume, spacing
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def centre_patch(volume: np.ndarray) -> np.ndarray:
    """
    Extracts a patch from the center of a volume.
    A real implementation would use a nodule coordinate to extract the patch.
    """
    c, h, w = volume.shape
    pc, ph, pw = PATCH_SIZE
    
    # Pad if volume is smaller than patch
    if c < pc or h < ph or w < pw:
        pad_c = max(0, pc - c)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        volume = np.pad(volume, ((0, pad_c), (0, pad_h), (0, pad_w)), mode='constant')
        c, h, w = volume.shape

    # Crop center
    return volume[c//2-pc//2 : c//2+pc//2, h//2-ph//2 : h//2+ph//2, w//2-pw//2 : w//2+pw//2]

def compute_gradcam(model, input_patch: np.ndarray):
    """
    Placeholder for 3D Grad-CAM computation.
    This would involve forward/backward passes to get gradients and activations.
    """
    print("--- Running placeholder Grad-CAM computation ---")
    # Simulate a heatmap as random noise for visualization purposes
    return np.random.rand(*input_patch.shape)

def render_gradcam_animation(patch: np.ndarray, heatmap: np.ndarray) -> dict:
    """
    Creates a high-resolution animated GIF scanning through the volume slices.
    Returns a dict with 'axial', 'coronal', 'sagittal' keys, each containing
    'raw' and 'overlay' base64 GIF strings.
    """
    depth, height, width = patch.shape
    # Upscale factor to make images significantly bigger (e.g., 5x)
    scale = 5
    
    # Containers for frames
    views = {
        "axial": {"raw": [], "overlay": []},
        "coronal": {"raw": [], "overlay": []},
        "sagittal": {"raw": [], "overlay": []}
    }

    def process_slice(slice_2d, heat_2d, label):
        h, w = slice_2d.shape
        new_size = (w * scale, h * scale)
        
        # Raw (Grayscale)
        img_raw = Image.fromarray(np.uint8(255 * slice_2d)).convert("RGB")
        img_raw = img_raw.resize(new_size, resample=Image.Resampling.NEAREST)
        
        # Overlay (Red Heatmap)
        slice_rgba = Image.fromarray(np.uint8(255 * slice_2d)).convert("RGBA")
        heatmap_rgba = np.zeros((*heat_2d.shape, 4), dtype=np.uint8)
        heatmap_rgba[..., 0] = 255  # Red
        heatmap_rgba[..., 3] = (heat_2d * 180).astype(np.uint8) # Alpha
        heat_img = Image.fromarray(heatmap_rgba)
        
        img_overlay = Image.alpha_composite(slice_rgba, heat_img).convert("RGB")
        img_overlay = img_overlay.resize(new_size, resample=Image.Resampling.NEAREST)
        
        # Add Labels
        for img in [img_raw, img_overlay]:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 14 * (scale // 2))
            except IOError:
                font = ImageFont.load_default()
            draw.text((12, 12), label, fill="black", font=font)
            draw.text((10, 10), label, fill="white", font=font)
            
        return img_raw, img_overlay

    for i in range(depth):
        # Axial (Z-axis)
        raw, over = process_slice(patch[i, :, :], heatmap[i, :, :], f"Axial Z={i}")
        views["axial"]["raw"].append(raw)
        views["axial"]["overlay"].append(over)
        
        # Coronal (Y-axis)
        raw, over = process_slice(patch[:, i, :], heatmap[:, i, :], f"Coronal Y={i}")
        views["coronal"]["raw"].append(raw)
        views["coronal"]["overlay"].append(over)

        # Sagittal (X-axis)
        raw, over = process_slice(patch[:, :, i], heatmap[:, :, i], f"Sagittal X={i}")
        views["sagittal"]["raw"].append(raw)
        views["sagittal"]["overlay"].append(over)

    # Convert all frame lists to base64 GIFs
    results = {}
    for view_name, type_dict in views.items():
        results[view_name] = {}
        for type_name, frames in type_dict.items():
            buffered = io.BytesIO()
            frames[0].save(buffered, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
            results[view_name][type_name] = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    return results