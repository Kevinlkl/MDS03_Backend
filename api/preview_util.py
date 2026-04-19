from io import BytesIO
import numpy as np
import torch
from PIL import Image


def tensor_middle_slice_to_png_bytes(
    tensor: torch.Tensor,
    normalize: bool = True,
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

    slice_uint8 = (slice_2d * 255).astype(np.uint8)

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer