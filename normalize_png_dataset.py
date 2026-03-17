import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tqdm

def improve_normalize_to_uint8(slice_data: np.ndarray, p_low: float=0.5, p_high: float=99.5) -> np.ndarray:
    """
    Enhanced normalization: Removes outliers, scales to 0-255, 
    and applies CLAHE for sharper anatomical detail.
    """
    # 1. Clean data and convert to float32
    arr = np.nan_to_num(slice_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Percentile-based clipping (Removes outlier bright/dark spots)
    # We only look at values > 0 to calculate percentiles (ignoring background)
    brain_voxels = arr[arr > 0]
    if brain_voxels.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    
    lo = float(np.percentile(brain_voxels, p_low))
    hi = float(np.percentile(brain_voxels, p_high))
    
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    
    # Clip and rescale to 0-1 range
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    
    # 3. Convert to uint8 (0-255)
    arr_uint8 = (arr * 255).astype(np.uint8)

    # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is the secret sauce for improving FID scores as it highlights
    # edges and textures that the Inception network (used by FID) looks for.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_arr = clahe.apply(arr_uint8)
    
    return enhanced_arr

def normalize_png_in_folder(root_dir: Path):
    """Iterates through a directory and applies the improved normalization."""
    png_files = list(root_dir.rglob('*.png'))
    if not png_files:
        print(f"No PNG files found in {root_dir}")
        return

    print(f"Found {len(png_files)} PNG files under {root_dir}")
    for png_path in tqdm.tqdm(png_files, desc=f"Enhancing {root_dir.name}"):
        try:
            # Open as grayscale ('L')
            img = Image.open(png_path).convert('L')
            arr = np.array(img)
            
            # Process
            norm_arr = improve_normalize_to_uint8(arr)
            
            # Save back to same path
            norm_img = Image.fromarray(norm_arr, mode='L')
            norm_img.save(png_path)
            
        except Exception as e:
            print(f"\nFailed to normalize {png_path}: {e}")

if __name__ == "__main__":
    # Your specific FYP directory structure
    base_dir = Path('Dataset/T1')
    subfolders = [
        'Glioma_256_T1weighted',
        'Meningioma_256_T1weighted',
        'Pituitary_256_T1weighted'
    ]
    
    print("--- Starting Enhanced Medical Image Normalization ---")
    for sub in subfolders:
        folder = base_dir / sub
        if folder.exists():
            print(f"\nProcessing folder: {folder}")
            normalize_png_in_folder(folder)
        else:
            print(f"\nWarning: Folder {folder} does not exist. Skipping.")
            
    print("\n--- Preprocessing Complete ---")