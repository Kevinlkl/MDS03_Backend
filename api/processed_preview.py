from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from model_t1_t2.config import Config
from model_t1_t2.preprocess import MRIProcessor
from api.preview_util import tensor_middle_slice_to_png_bytes

router = APIRouter(prefix="/api", tags=["Processed Preview"])

processor = MRIProcessor(
    spatial_size=Config.SPATIAL_SIZE,
    pixdim=Config.PIXDIM,
    source_key=Config.INPUT_KEY,
    intensity_lower=Config.INTENSITY_LOWER,
    intensity_upper=Config.INTENSITY_UPPER,
    b_min=Config.B_MIN,
    b_max=Config.B_MAX,
)

@router.post("/preview-preprocessed-t1")
async def preview_preprocessed_t1(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        item = processor.preprocess(input_path, device=Config.DEVICE)
        t1 = item["image"]  # same tensor used by inference pipeline

        png_buffer = tensor_middle_slice_to_png_bytes(t1)

        return StreamingResponse(png_buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preprocessed T1 preview: {str(e)}")
    
@router.post("/preview-preprocessed-t2")
async def preview_preprocessed_t2(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        item = processor.preprocess(input_path, device=Config.DEVICE)
        t2 = item["image"]

        png_buffer = tensor_middle_slice_to_png_bytes(t2)

        return StreamingResponse(png_buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preprocessed T2 preview: {str(e)}")