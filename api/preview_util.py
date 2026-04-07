from io import BytesIO
import numpy as np
import torch
from PIL import Image


def tensor_middle_slice_to_png_bytes(
    tensor: torch.Tensor,
    normalize: bool = True,
) -> BytesIO:
    """
    Expects tensor shape like:
    [1, 1, H, W, D] or [1, H, W, D] or [H, W, D]
    Returns PNG bytes of the middle axial slice.
    """
    arr = tensor.detach().cpu().float().numpy()

    # Remove batch/channel dims if present
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeezing, got shape {arr.shape}")

    mid = arr.shape[2] // 2
    slice_2d = arr[:, :, mid]

    slice_2d = np.nan_to_num(slice_2d)

    if normalize:
        min_val = float(slice_2d.min())
        max_val = float(slice_2d.max())
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val)
        else:
            slice_2d = np.zeros_like(slice_2d)

    slice_uint8 = (slice_2d * 255).astype(np.uint8)

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer