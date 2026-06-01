"""Shared preview helpers for converting model tensors into PNG images."""

from io import BytesIO
import numpy as np
import torch
from PIL import Image


def tensor_middle_slice_to_png_bytes(
    tensor: torch.Tensor,
    normalize: bool = True,
) -> BytesIO:
    """
    Function description:
        Render the middle axial slice of a tensor volume as PNG bytes.

    Parameters:
        tensor (torch.Tensor): Tensor volume to preview.
        normalize (bool): Whether to map model output values from [-1, 1] to [0, 1].

    Returns:
        BytesIO: In-memory PNG image buffer positioned at the beginning.
    """
    # Detach from autograd and move to CPU so NumPy/PIL can process the data.
    arr = tensor.detach().cpu().float().numpy()

    # Model tensors often include batch/channel dimensions; keep the first item.
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

    mid = arr.shape[2] // 2
    slice_2d = arr[:, :, mid]

    # Replace NaN/Inf values so image conversion does not fail.
    slice_2d = np.nan_to_num(slice_2d)

    if normalize:
        # Model outputs are expected in [-1, 1]; map that range to [0, 1].
        slice_2d = np.clip(slice_2d, -1.0, 1.0)
        slice_2d = (slice_2d + 1.0) / 2.0

    # PIL expects uint8 pixel values for a standard grayscale PNG.
    slice_uint8 = (slice_2d * 255).astype(np.uint8)

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer
