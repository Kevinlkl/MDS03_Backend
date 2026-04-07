from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from model_t1_t2.inference import InferencePipeline

router = APIRouter(prefix="/api", tags=["T1 to T2 Inference"])

pipeline = InferencePipeline()


def cleanup_files(*paths):
    for p in paths:
        try:
            path = Path(p)
            if path.exists():
                path.unlink()
        except Exception:
            pass


@router.post("/infer_t1_t2")
async def infer_mri(
    file: UploadFile = File(...),
    num_inference_steps: int = Form(1000),
):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported."
        )

    if not (1 <= num_inference_steps <= 1000):
        raise HTTPException(
            status_code=400,
            detail="num_inference_steps must be between 1 and 1000."
        )

    try:
        suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_output:
            output_path = tmp_output.name

        pipeline.run(
            input_path=input_path,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
        )

        if file.filename.endswith(".nii.gz"):
            base_name = file.filename[:-7]
        else:
            base_name = Path(file.filename).stem

        download_name = f"{base_name}_pred_t2_{num_inference_steps}steps{suffix}"

        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename=download_name,
            background=BackgroundTask(cleanup_files, input_path, output_path),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")