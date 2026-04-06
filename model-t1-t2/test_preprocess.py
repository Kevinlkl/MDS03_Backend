# test_preprocess.py

from config import Config
from preprocess import MRIProcessor

processor = MRIProcessor(
    spatial_size=Config.SPATIAL_SIZE,
    pixdim=Config.PIXDIM,
    source_key=Config.INPUT_KEY,
    intensity_lower=Config.INTENSITY_LOWER,
    intensity_upper=Config.INTENSITY_UPPER,
    b_min=Config.B_MIN,
    b_max=Config.B_MAX,
)

sample_path = Config.UPLOADS_DIR / "sample_t1.nii"

result = processor.preprocess(sample_path, device=Config.DEVICE)

print("Processed tensor shape:", result["image"].shape)
print("Original path:", result["original_path"])
print("Metadata keys:", result["meta"].keys())