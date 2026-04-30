from io import BytesIO
import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def _apply_clahe(image_2d: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE) for improved contrast.
    
    Args:
        image_2d: 2D grayscale image (0-255 uint8)
        clip_limit: Clipping limit for histogram (higher = more contrast)
        grid_size: Divide image into grid_size x grid_size tiles
    
    Returns:
        Contrast-enhanced 2D image
    """
    from skimage import exposure
    
    # Ensure uint8 for CLAHE
    img_uint8 = np.clip(image_2d * 255, 0, 255).astype(np.uint8)
    
    # Apply CLAHE
    enhanced = exposure.equalize_adapthist(img_uint8, clip_limit=clip_limit / 100.0)
    
    return enhanced


def tensor_middle_slice_to_png_bytes(
    tensor: torch.Tensor,
    normalize: bool = True,
    enhance_contrast: bool = True,
) -> BytesIO:
    arr = tensor.detach().cpu().float().numpy()

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

    mid = arr.shape[2] // 2
    slice_2d = arr[:, :, mid]

    slice_2d = np.nan_to_num(slice_2d)

    if normalize:
        # FIXED normalization for model outputs
        slice_2d = np.clip(slice_2d, -1.0, 1.0)
        slice_2d = (slice_2d + 1.0) / 2.0
    else:
        # If not normalizing, ensure values are in [0, 1] range
        vmin, vmax = slice_2d.min(), slice_2d.max()
        if vmax > vmin:
            slice_2d = (slice_2d - vmin) / (vmax - vmin)
        else:
            slice_2d = np.zeros_like(slice_2d)
    
    # Convert to 0-255 range before contrast enhancement
    slice_uint8 = (slice_2d * 255).astype(np.uint8)
    
    # Apply contrast enhancement to improve clarity
    if enhance_contrast:
        try:
            slice_enhanced = _apply_clahe(slice_uint8 / 255.0, clip_limit=3.0)
            slice_uint8 = (slice_enhanced * 255).astype(np.uint8)
        except Exception as e:
            print(f"Warning: CLAHE failed ({e}), using original slice")

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer