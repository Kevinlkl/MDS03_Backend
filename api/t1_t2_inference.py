from pathlib import Path
import tempfile
import shutil
import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from api.preview_util import tensor_middle_slice_to_png_bytes

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

def png_buffer_to_base64(png_buffer) -> str:
    return base64.b64encode(png_buffer.getvalue()).decode("utf-8")

@router.post("/infer_t1_t2")
async def infer_mri(
    file: UploadFile = File(...),
    ground_truth_file: Optional[UploadFile] = File(None),
    num_inference_steps: int = Form(1000),
):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported for input file.",
        )

    if ground_truth_file and not ground_truth_file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported for ground truth file.",
        )

    if not (1 <= num_inference_steps <= 1000):
        raise HTTPException(
            status_code=400,
            detail="num_inference_steps must be between 1 and 1000.",
        )

    input_path = None
    gt_path = None
    output_path = None

    try:
        input_suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        if ground_truth_file is not None:
            gt_suffix = ".nii.gz" if ground_truth_file.filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=gt_suffix) as tmp_gt:
                shutil.copyfileobj(ground_truth_file.file, tmp_gt)
                gt_path = tmp_gt.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_output:
            output_path = tmp_output.name

        result = pipeline.run_and_evaluate(
            input_path=input_path,
            gt_path=gt_path,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
        )

        t1 = result["t1"]
        pred_t2 = result["pred_t2"]
        gt_t2 = result["gt_t2"]
        metrics = result["metrics"]

        t1_preview_b64 = png_buffer_to_base64(
            tensor_middle_slice_to_png_bytes(t1)
        )
        pred_preview_b64 = png_buffer_to_base64(
            tensor_middle_slice_to_png_bytes(pred_t2)
        )
        gt_preview_b64 = (
            png_buffer_to_base64(tensor_middle_slice_to_png_bytes(gt_t2))
            if gt_t2 is not None
            else None
        )

        if file.filename.endswith(".nii.gz"):
            base_name = file.filename[:-7]
        else:
            base_name = Path(file.filename).stem

        download_name = f"{base_name}_pred_t2_{num_inference_steps}steps.nii.gz"

        return {
            "success": True,
            "output_path": result["output_path"],
            "download_name": download_name,
            "has_ground_truth": gt_t2 is not None,
            "metrics": {
                "psnr": round(float(metrics["psnr"]), 4) if metrics["psnr"] is not None else None,
                "ssim": round(float(metrics["ssim"]), 4) if metrics["ssim"] is not None else None,
                "fid": round(float(metrics["fid"]), 4) if metrics["fid"] is not None else None,
            },
            "previews": {
                "input": t1_preview_b64,
                "generated": pred_preview_b64,
                "ground_truth": gt_preview_b64,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        # keep output file if you want to download it later
        # remove input / gt temp files only
        cleanup_files(*(p for p in [input_path, gt_path] if p is not None))


@router.get("/download_t1_t2")
async def download_t1_t2(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=str(p),
        media_type="application/octet-stream",
        filename=p.name,
    )