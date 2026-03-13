# Stable Diffusion LoRA Training for T1 Brain Tumor MRI

This project now includes a LoRA-first training and generation CLI in `model.py`.

## 1. Environment and Dependencies

From workspace root:

```powershell
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -r requirements-sd.txt
```

Verify CUDA in PyTorch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 2. Train LoRA from a Prebuilt Stable Diffusion Model

Default dataset path expects:

- `Dataset/T1/Glioma_256_T1weighted`
- `Dataset/T1/Meningioma_256_T1weighted`
- `Dataset/T1/Pituitary_256_T1weighted`

Run LoRA training:

```powershell
python model.py train --data-root Dataset/T1 --output-dir outputs/sd_t1_mri_lora --base-model runwayml/stable-diffusion-v1-5 --train-method lora --epochs 3 --batch-size 2 --image-size 256 --learning-rate 1e-4 --fp16
```

Notes:

- LoRA is much faster and more stable than full UNet fine-tuning.
- First run downloads pretrained weights from Hugging Face.
- Checkpoints are saved every 500 steps by default.

## 3. Generate Synthetic T1 MRI Images with LoRA

```powershell
python model.py generate --base-model runwayml/stable-diffusion-v1-5 --lora-path outputs/sd_t1_mri_lora/lora --output-dir outputs/generated --prompt "axial T1-weighted brain MRI scan showing glioma, medical imaging, grayscale" --num-images 8 --image-size 256 --fp16
```

## 4. Tips for Better Medical Realism

- Use more specific prompts such as: `post-contrast axial T1-weighted brain MRI showing pituitary tumor`.
- Increase epochs gradually and monitor quality to avoid overfitting.
- For full model fine-tuning, use `--train-method full`.
