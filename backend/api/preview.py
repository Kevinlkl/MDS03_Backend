"""Raw NIfTI preview endpoints.

This router returns a PNG of the middle slice from an uploaded MRI volume. It is
useful for quick visual checks before or after inference.
"""

from io import BytesIO
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

import nibabel as nib
import numpy as np
from PIL import Image

router = APIRouter(prefix="/api", tags=["Preview"])


def get_middle_slice_png_bytes(nifti_path: str, axis: int = 2, volume_index: int = 0) -> BytesIO:
    """
    Function description:
        Load a NIfTI file and render one middle slice as PNG bytes.

    Parameters:
        nifti_path (str): Path to the NIfTI file to preview.
        axis (int): Axis used to choose the middle slice.
        volume_index (int): Volume index to use when the NIfTI file is 4D.

    Returns:
        BytesIO: In-memory PNG image buffer positioned at the beginning.
    """
    # Nibabel reads the volume data into a NumPy array while preserving dimensions.
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # 4D NIfTI files may contain multiple volumes; select one before slicing.
    if data.ndim == 4:
        data = data[..., volume_index]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}")

    mid = data.shape[axis] // 2

    # Slice along the requested anatomical axis.
    if axis == 0:
        slice_2d = data[mid, :, :]
    elif axis == 1:
        slice_2d = data[:, mid, :]
    else:
        slice_2d = data[:, :, mid]

    slice_2d = np.nan_to_num(slice_2d)

    # Normalize the selected slice independently so low-contrast scans preview well.
    min_val = float(slice_2d.min())
    max_val = float(slice_2d.max())

    if max_val > min_val:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)
    else:
        slice_2d = np.zeros_like(slice_2d, dtype=np.float32)

    # Convert normalized [0, 1] intensities into an 8-bit grayscale PNG.
    slice_uint8 = (slice_2d * 255).astype(np.uint8)

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


@router.post("/preview-middle-slice")
async def preview_middle_slice(file: UploadFile = File(...)):
    """
    Function description:
        Return a PNG preview of the middle slice from an uploaded NIfTI volume.

    Parameters:
        file (UploadFile): Uploaded .nii or .nii.gz MRI volume.

    Returns:
        StreamingResponse: PNG image response for the selected middle slice.
    """
    # Reject non-NIfTI files early because nibabel errors are less user-friendly.
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported."
        )

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

    try:
        # Write the upload to disk because nibabel expects a file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        # Stream the PNG directly so the client can display it as an image response.
        png_buffer = get_middle_slice_png_bytes(input_path)

        return StreamingResponse(
            png_buffer,
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")
