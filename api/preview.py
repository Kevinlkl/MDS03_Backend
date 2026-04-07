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
    img = nib.load(nifti_path)
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., volume_index]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}")

    mid = data.shape[axis] // 2

    if axis == 0:
        slice_2d = data[mid, :, :]
    elif axis == 1:
        slice_2d = data[:, mid, :]
    else:
        slice_2d = data[:, :, mid]

    slice_2d = np.nan_to_num(slice_2d)

    min_val = float(slice_2d.min())
    max_val = float(slice_2d.max())

    if max_val > min_val:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)
    else:
        slice_2d = np.zeros_like(slice_2d, dtype=np.float32)

    slice_uint8 = (slice_2d * 255).astype(np.uint8)

    image = Image.fromarray(slice_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


@router.post("/preview-middle-slice")
async def preview_middle_slice(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported."
        )

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        png_buffer = get_middle_slice_png_bytes(input_path)

        return StreamingResponse(
            png_buffer,
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")