# T1 Synthetic FID Evaluation - Updated Documentation

## Overview
The T1 synthesis FID evaluation has been upgraded from a simplified approximation to proper **Fréchet Inception Distance (FID)** computation using InceptionV3 features, matching the production-quality evaluation used in postprocess.py.

## Key Changes

### 1. **postprocess.py** (New)
Created a dedicated postprocessing module with proper FID computation:
- InceptionV3-based feature extraction
- Gaussian statistics computation
- Proper Fréchet Distance calculation
- Google Drive T1 dataset reference support

### 2. **compute_fid_score()** (Updated)
Enhanced with optional ground truth support:

```python
def compute_fid_score(
    pred_tensor: torch.Tensor,
    gt_tensor: Optional[torch.Tensor] = None,
) -> Optional[float]:
```

**Behavior:**
- With `gt_tensor`: Uses proper FID with InceptionV3 features
- Without `gt_tensor`: Falls back to lightweight approximation (fast, no reference needed)

### 3. **Pipeline Methods** (New)
Added to `SyntheticT1GenerationPipeline`:

#### `compute_fid_with_reference()`
```python
fid = pipeline.compute_fid_with_reference(
    generated_path="/path/to/generated_t1.nii",
    gt_tensor=ground_truth_tensor
)
```
Computes FID between generated and reference T1 data.

#### `compute_fid_from_gdrive()`
```python
fid = pipeline.compute_fid_from_gdrive(
    gdrive_zip_link="https://drive.google.com/...",
    generated_path="/path/to/generated_t1.nii",
    cache_dir="/optional/cache"
)
```
Computes FID using T1 reference data from Google Drive ZIP archive.

## Usage Examples

### Example 1: Quick Per-Sample FID (No Reference)
```python
# Fast approximation for each generated sample
fid_score = compute_fid_score(generated_tensor)
print(f"FID Score: {fid_score:.2f}")  # Range: 0-200 (lower is better)
```

### Example 2: Proper FID with Ground Truth
```python
# Proper FID computation with reference data
fid_score = compute_fid_score(
    pred_tensor=generated_t1,
    gt_tensor=ground_truth_t1
)
print(f"FID Score: {fid_score:.4f}")  # More precise value
```

### Example 3: Batch FID Computation
```python
from model_t1_synthetic.postprocess import compute_fid_from_tensors

# Generate multiple samples
results = pipeline.generate_many(num_samples=10, seed=42)

# Load all predictions and GT
pred_batch = torch.stack([r["tensor"] for r in results])
gt_batch = load_ground_truth_batch()

# Compute FID for the entire batch
fid = compute_fid_from_tensors(
    pred_t1=pred_batch,
    gt_t1=gt_batch,
    device="cuda",
    batch_size=16,
    max_slices_per_volume=64
)
print(f"Batch FID: {fid:.4f}")
```

### Example 4: FID with Google Drive Reference
```python
# Setup: Your T1 reference data in a Google Drive ZIP with structure:
# dataset.zip
# └── T1/
#     ├── subject_001_t1.nii
#     ├── subject_002_t1.nii
#     └── ...

gdrive_link = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"

fid = pipeline.compute_fid_from_gdrive(
    gdrive_zip_link=gdrive_link,
    generated_path="/path/to/generated_t1.nii",
    cache_dir="/optional/persistent/cache"
)
print(f"FID vs Google Drive T1 Reference: {fid:.4f}")
```

## API Integration Example

Update your FastAPI endpoint to support optional reference data:

```python
from fastapi import FastAPI, File, UploadFile
from model_t1_synthetic.inference import SyntheticT1GenerationPipeline

app = FastAPI()
pipeline = None

@app.post("/api/generate_synthetic_t1")
async def generate_with_fid(
    num_samples: int,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    gdrive_reference: Optional[str] = None
):
    """Generate T1 samples and optionally compute FID against reference."""
    global pipeline
    if pipeline is None:
        pipeline = SyntheticT1GenerationPipeline()
    
    # Generate samples
    results = pipeline.generate_many(
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    # Optional: Compute FID with reference
    fid_scores = []
    if gdrive_reference:
        for result in results:
            try:
                fid = pipeline.compute_fid_from_gdrive(
                    gdrive_zip_link=gdrive_reference,
                    generated_path=result["output_path"],
                    cache_dir="/tmp/fid_cache"
                )
                fid_scores.append(fid)
            except Exception as e:
                print(f"FID computation failed: {e}")
                fid_scores.append(None)
    
    return {
        "success": True,
        "num_samples": num_samples,
        "generated_files": results,
        "fid_scores": fid_scores if fid_scores else None
    }
```

## FID Score Interpretation

| FID Range | Quality | Interpretation |
|-----------|---------|-----------------|
| 0-30 | Excellent | Generated samples very similar to reference |
| 30-60 | Good | High quality, minor differences |
| 60-100 | Fair | Acceptable quality, noticeable differences |
| 100-150 | Poor | Significant quality issues |
| 150+ | Very Poor | Major artifacts or mode collapse |

*Note: These ranges are approximate. Reference dataset quality significantly affects FID values.*

## Dependencies

Requires these packages (already in requirements):
- `torch`
- `torchvision` (for InceptionV3)
- `nibabel` (for NIfTI I/O)
- `numpy`
- `pillow` (for image conversion)
- `gdown` (optional, for automatic Google Drive downloads)

## Performance Considerations

### Lightweight FID (No Reference)
- **Speed**: <10ms per sample
- **Memory**: Minimal (~10MB)
- **Accuracy**: Approximation only
- **Use Case**: Quick per-sample quality check

### Proper FID (With Reference)
- **Speed**: ~2-5 seconds per comparison (depending on volume size)
- **Memory**: ~4-8GB GPU memory
- **Accuracy**: True FID metric
- **Use Case**: Publication-quality evaluation

### Google Drive FID
- **Speed**: Same as proper FID + download time (~10-30 minutes for first run)
- **Memory**: Same as proper FID
- **Caching**: Automatic via `cache_dir` parameter
- **Use Case**: Reproducible evaluation against shared reference dataset

## Backward Compatibility

The updated `compute_fid_score()` maintains backward compatibility:
```python
# Old code still works (uses approximation)
fid = compute_fid_score(tensor)

# New code with ground truth (uses proper FID)
fid = compute_fid_score(tensor, gt_tensor)
```

## References
- [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500)
- [InceptionV3 Features](https://arxiv.org/abs/1512.00567)
- [Medical Image Generation Metrics](https://ieeexplore.ieee.org/document/9521244)
