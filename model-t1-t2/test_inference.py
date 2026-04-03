from config import Config
from inference import InferencePipeline

pipeline = InferencePipeline()

input_path = Config.UPLOADS_DIR / "sample_t1.nii"
output_path = Config.OUTPUTS_DIR / "sample_t1_pred_t2.nii"

result = pipeline.run(str(input_path), str(output_path))
print("Saved output to:", result)