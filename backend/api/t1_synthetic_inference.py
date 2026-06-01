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
from model_t1_synthetic.inference import (
    SyntheticT1GenerationPipeline,
    zip_generated_files,
)

router = APIRouter(prefix="/api", tags=["Synthetic T1 Generation"])

_pipeline: Optional[SyntheticT1GenerationPipeline] = None


class SyntheticGenerationRequest(BaseModel):
    """Request body for synthetic T1 batch generation."""

    num_samples: int = 1
    num_inference_steps: int = Config.NUM_INFERENCE_STEPS
    seed: Optional[int] = None


def get_pipeline() -> SyntheticT1GenerationPipeline:
    """Create and cache the synthetic T1 generation pipeline on first use."""
    global _pipeline
    if _pipeline is None:
        # Defer checkpoint loading until this endpoint is called.
        _pipeline = SyntheticT1GenerationPipeline()
    return _pipeline


def tensor_to_base64_png(tensor) -> str:
    """Convert the middle slice of a tensor volume into a base64 PNG string."""
    png_buffer = tensor_middle_slice_to_png_bytes(tensor)
    return base64.b64encode(png_buffer.getvalue()).decode("utf-8")


@router.post("/generate_synthetic_t1")
async def generate_synthetic_t1(payload: SyntheticGenerationRequest):
    """Generate one or more synthetic T1 MRI volumes and return previews."""
    num_samples = payload.num_samples
    num_inference_steps = payload.num_inference_steps
    seed = payload.seed

    # Keep request bounds small enough for an interactive API call.
    if not 1 <= num_samples <= 100:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 100.",
        )

    if not 1 <= num_inference_steps <= 1000:
        raise HTTPException(
            status_code=400,
            detail="num_inference_steps must be between 1 and 1000.",
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Config.GENERATED_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Pipeline construction loads checkpoints lazily.
        pipeline = get_pipeline()

        generated = pipeline.generate_many(
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_dir=batch_dir,
        )

        if not generated:
            raise RuntimeError("No synthetic T1 samples were generated.")

        archive_path = Config.GENERATED_DIR / f"synthetic_t1_{timestamp}.zip"

        # Package all generated NIfTI files into one downloadable archive.
        zip_generated_files(
            [item["output_path"] for item in generated],
            archive_path,
        )

        first_generated = generated[0]
        output_path = str(first_generated["output_path"])

        generated_preview = tensor_to_base64_png(first_generated["tensor"])

        generated_files = []

        for item in generated:
            preview_base64 = tensor_to_base64_png(item["tensor"])

            generated_files.append(
                {
                    "index": item["index"],
                    "seed": item["seed"],
                    "output_path": item["output_path"],
                    "preview": preview_base64,
                    "middle_slice_preview": preview_base64,
                }
            )

        response = {
            "success": True,
            "mode": "synthetic-t1",
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "archive_path": str(archive_path),
            "download_name": archive_path.name,
            "output_path": output_path,
            "has_ground_truth": False,
            "metrics": {
                "psnr": None,
                "ssim": None,
                "dataset_mean_psnr": None,
                "dataset_mean_ssim": None,
                "dataset_fid": Config.PRECOMPUTED_FID,
                "dataset_kid_mean": Config.PRECOMPUTED_KID_MEAN,
                "dataset_kid_std": Config.PRECOMPUTED_KID_STD,
                "dataset_metric_scope": (
                    "Precomputed full evaluation dataset metrics, not recalculated per batch"
                ),
            },
            "previews": {
                "input": None,
                "ground_truth": None,
                "generated": generated_preview,
            },
            "generated_files": generated_files,
        }

        print("Synthetic T1 response keys:", list(response.keys()))

        return response

    except Exception as exc:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Synthetic generation failed: "
                f"{type(exc).__name__}: {str(exc)}"
            ),
        ) from exc


@router.get("/download_synthetic_t1")
async def download_synthetic_t1(path: str):
    """Download a ZIP archive created by the synthetic T1 generation endpoint."""
    archive_path = Path(path)

    if not archive_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Archive not found.",
        )

    return FileResponse(
        path=str(archive_path),
        media_type="application/zip",
        filename=archive_path.name,
    )
