# Stable Diffusion Training for T1 Brain Tumor MRI

This project now includes a training and generation CLI in `model.py`.

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

## 2. Train from a Prebuilt Stable Diffusion Model

Default dataset path expects:

- `Dataset/T1/Glioma_256_T1weighted`
- `Dataset/T1/Meningioma_256_T1weighted`
- `Dataset/T1/Pituitary_256_T1weighted`

Run training:

```powershell
python model.py train --data-root Dataset/T1 --output-dir outputs/sd_t1_mri --base-model runwayml/stable-diffusion-v1-5 --epochs 5 --batch-size 2 --image-size 256 --fp16
```

Notes:

- GPU CUDA is required for training.
- First run downloads pretrained weights from Hugging Face.
- Checkpoints are saved every 500 steps by default.

## 3. Generate Synthetic T1 MRI Images

```powershell
python model.py generate --model-path outputs/sd_t1_mri --output-dir outputs/generated --prompt "axial T1-weighted brain MRI scan showing glioma, medical imaging, grayscale" --num-images 8 --image-size 256 --fp16
```

## 4. Tips for Better Medical Realism

- Use more specific prompts such as: `post-contrast axial T1-weighted brain MRI showing pituitary tumor`.
- Increase epochs gradually and monitor quality to avoid overfitting.
- Consider using LoRA or domain-specific diffusion checkpoints if full fine-tuning becomes unstable.
