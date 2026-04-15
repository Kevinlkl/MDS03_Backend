from model_t1_t2.config import Config
from model_t1_t2.inference import InferencePipeline

pipeline = InferencePipeline()

input_path = Config.UPLOADS_DIR / "sample_t1.nii"
output_path = Config.OUTPUTS_DIR / "sample_t1_pred_t2.nii"
gt_path = Config.GT_DIR / "sample_t2_gt.nii"

result = pipeline.run_and_evaluate(
    input_path=input_path,
    gt_path=gt_path,
    num_inference_steps=1000
)

print("Output saved at:", result["output_path"])
print("PSNR:", result["metrics"]["psnr"])
print("SSIM:", result["metrics"]["ssim"])
