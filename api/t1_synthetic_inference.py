from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.preview_util import tensor_middle_slice_to_png_bytes
from model_t1_synthetic.config import Config
from model_t1_synthetic.inference import SyntheticT1GenerationPipeline, zip_generated_files, compute_fid_score

router = APIRouter(prefix="/api", tags=["Synthetic T1 Generation"])

_pipeline: Optional[SyntheticT1GenerationPipeline] = None


class SyntheticGenerationRequest(BaseModel):
    num_samples: int = 1
    num_inference_steps: int = Config.NUM_INFERENCE_STEPS
    seed: Optional[int] = None


def get_pipeline() -> SyntheticT1GenerationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SyntheticT1GenerationPipeline()
    return _pipeline


def tensor_to_base64_png(tensor) -> str:
    return base64.b64encode(tensor_middle_slice_to_png_bytes(tensor).getvalue()).decode("utf-8")


@router.post("/generate_synthetic_t1")
async def generate_synthetic_t1(payload: SyntheticGenerationRequest):
    num_samples = payload.num_samples
    num_inference_steps = payload.num_inference_steps
    seed = payload.seed

    if not 1 <= num_samples <= 100:
        raise HTTPException(status_code=400, detail="num_samples must be between 1 and 100.")

    if not 1 <= num_inference_steps <= 1000:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 1000.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Config.GENERATED_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = get_pipeline()
        generated = pipeline.generate_many(
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_dir=batch_dir,
        )

        archive_path = Config.GENERATED_DIR / f"synthetic_t1_{timestamp}.zip"
        zip_generated_files([item["output_path"] for item in generated], archive_path)

        return {
            "success": True,
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "archive_path": str(archive_path),
            "download_name": archive_path.name,
            "generated_files": [
                {
                    "index": item["index"],
                    "seed": item["seed"],
                    "output_path": item["output_path"],
                    "preview": tensor_to_base64_png(item["tensor"]),
                    "middle_slice_preview": tensor_to_base64_png(item["tensor"]),
                    "fid_score": compute_fid_score(item["tensor"]),
                }
                for item in generated
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthetic generation failed: {str(exc)}") from exc


@router.get("/download_synthetic_t1")
async def download_synthetic_t1(path: str):
    archive_path = Path(path)
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Archive not found.")

    return FileResponse(
        path=str(archive_path),
        media_type="application/zip",
        filename=archive_path.name,
    )
