import os
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm

def normalize_to_uint8(slice_data: np.ndarray, p_low: float=0.5, p_high: float=99.5) -> np.ndarray:
    arr = np.nan_to_num(slice_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    brain_voxels = arr[arr > 0]
    if brain_voxels.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(brain_voxels, p_low))
    hi = float(np.percentile(brain_voxels, p_high))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)

def normalize_png_in_folder(root_dir: Path):
    png_files = list(root_dir.rglob('*.png'))
    print(f"Found {len(png_files)} PNG files under {root_dir}")
    for png_path in tqdm.tqdm(png_files, desc="Normalizing PNGs"):
        try:
            img = Image.open(png_path).convert('L')
            arr = np.array(img)
            norm_arr = normalize_to_uint8(arr)
            norm_img = Image.fromarray(norm_arr, mode='L')
            norm_img.save(png_path)
        except Exception as e:
            print(f"Failed to normalize {png_path}: {e}")

if __name__ == "__main__":
    # Automatically normalize PNGs in three folders
    base_dir = Path('Dataset/T1')
    subfolders = [
        'Glioma_256_T1weighted',
        'Meningioma_256_T1weighted',
        'Pituitary_256_T1weighted'
    ]
    for sub in subfolders:
        folder = base_dir / sub
        print(f"\nProcessing folder: {folder}")
        normalize_png_in_folder(folder)
