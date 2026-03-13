import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
	DDPMScheduler,
	StableDiffusionPipeline,
	UNet2DConditionModel,
	AutoencoderKL,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def class_name_from_folder(folder_name: str) -> str:
	base = folder_name.lower().replace("_256_t1weighted", "")
	if "glioma" in base:
		return "glioma"
	if "meningioma" in base:
		return "meningioma"
	if "pituitary" in base:
		return "pituitary tumor"
	return base.replace("_", " ")


class T1TumorPromptDataset(Dataset):
	def __init__(self, data_root: Path, image_size: int = 256) -> None:
		self.data_root = Path(data_root)
		self.samples: list[tuple[Path, str]] = []
		self.transforms = transforms.Compose(
			[
				transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
			]
		)

		if not self.data_root.exists():
			raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

		class_dirs = [p for p in self.data_root.iterdir() if p.is_dir()]
		if not class_dirs:
			raise ValueError(f"No class folders found under: {self.data_root}")

		for class_dir in sorted(class_dirs):
			tumor_name = class_name_from_folder(class_dir.name)
			prompt = f"axial T1-weighted brain MRI scan showing {tumor_name}, medical imaging, grayscale"

			for img_path in class_dir.rglob("*"):
				if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
					self.samples.append((img_path, prompt))

		if not self.samples:
			raise ValueError(f"No image files found under: {self.data_root}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
		path, prompt = self.samples[idx]
		image = Image.open(path).convert("L")
		image = image.convert("RGB")
		image = self.transforms(image)
		return {"pixel_values": image, "prompt": prompt}


def collate_batch(examples: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
	pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
	prompts = [ex["prompt"] for ex in examples]
	return {"pixel_values": pixel_values, "prompts": prompts}


def train(args: argparse.Namespace) -> None:
	seed_everything(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type != "cuda":
		raise RuntimeError("CUDA GPU is required for Stable Diffusion training.")

	data_root = Path(args.data_root)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	dataset = T1TumorPromptDataset(data_root=data_root, image_size=args.image_size)
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		collate_fn=collate_batch,
	)

	tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
	vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
	unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
	noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

	text_encoder.requires_grad_(False)
	vae.requires_grad_(False)
	unet.train()

	if args.train_method == "lora":
		unet.requires_grad_(False)
		lora_config = LoraConfig(
			r=args.lora_rank,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			init_lora_weights="gaussian",
			target_modules=["to_q", "to_k", "to_v", "to_out.0"],
		)
		unet.add_adapter(lora_config)

	text_encoder.to(device)
	vae.to(device)
	unet.to(device)

	use_amp = args.fp16 and device.type == "cuda"
	if args.fp16 and device.type != "cuda":
		print("Warning: --fp16 is enabled but CUDA is unavailable. Training will run in float32.")

	dtype = torch.float32

	params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
	if not params_to_optimize:
		raise RuntimeError("No trainable parameters found. Check training configuration.")

	optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	trainable_param_count = sum(p.numel() for p in params_to_optimize)
	print(f"Training method: {args.train_method}")
	print(f"Trainable parameters: {trainable_param_count:,}")

	total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
	progress_bar = tqdm(total=total_steps, desc="Training")
	global_step = 0

	for epoch in range(args.epochs):
		for batch in dataloader:
			pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
			prompts = batch["prompts"]

			with torch.no_grad():
				with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
					latents = vae.encode(pixel_values).latent_dist.sample()
					latents = latents * vae.config.scaling_factor

			noise = torch.randn_like(latents)
			timesteps = torch.randint(
				low=0,
				high=noise_scheduler.config.num_train_timesteps,
				size=(latents.shape[0],),
				device=device,
				dtype=torch.long,
			)

			noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

			tokenized = tokenizer(
				prompts,
				padding="max_length",
				truncation=True,
				max_length=tokenizer.model_max_length,
				return_tensors="pt",
			)
			input_ids = tokenized.input_ids.to(device)

			with torch.no_grad():
				with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
					encoder_hidden_states = text_encoder(input_ids)[0]

			with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
				model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
				loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

			if not torch.isfinite(loss):
				print(f"Warning: non-finite loss ({loss.item()}) at step {global_step + 1}. Skipping this batch.")
				optimizer.zero_grad(set_to_none=True)
				continue

			optimizer.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			global_step += 1
			progress_bar.update(1)
			progress_bar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

			if args.checkpoint_steps > 0 and global_step % args.checkpoint_steps == 0:
				ckpt_dir = output_dir / f"checkpoint-{global_step}"
				ckpt_dir.mkdir(parents=True, exist_ok=True)
				if args.train_method == "lora":
					unet.save_attn_procs(ckpt_dir / "lora")
				else:
					unet.save_pretrained(ckpt_dir / "unet")

	progress_bar.close()

	if args.train_method == "lora":
		lora_dir = output_dir / "lora"
		lora_dir.mkdir(parents=True, exist_ok=True)
		unet.save_attn_procs(lora_dir)
		(output_dir / "base_model.txt").write_text(args.base_model, encoding="utf-8")
		print(f"LoRA adapter saved to: {lora_dir}")
		print(f"Base model recorded in: {output_dir / 'base_model.txt'}")
	else:
		pipe = StableDiffusionPipeline.from_pretrained(
			args.base_model,
			vae=vae,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			safety_checker=None,
		)
		pipe.save_pretrained(output_dir)
		print(f"Model saved to: {output_dir}")


@torch.inference_mode()
def generate(args: argparse.Namespace) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.lora_path:
		pipe = StableDiffusionPipeline.from_pretrained(args.base_model, safety_checker=None)
		pipe.unet.load_attn_procs(args.lora_path)
	else:
		model_path = Path(args.model_path)
		pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
	pipe = pipe.to(device)
	if args.fp16 and device.type == "cuda":
		pipe = pipe.to(dtype=torch.float16)

	for i in range(args.num_images):
		image = pipe(
			prompt=args.prompt,
			negative_prompt=args.negative_prompt,
			guidance_scale=args.guidance_scale,
			num_inference_steps=args.inference_steps,
			height=args.image_size,
			width=args.image_size,
		).images[0]
		image.save(output_dir / f"sample_{i + 1:03d}.png")

	print(f"Generated {args.num_images} images in: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train/generate Stable Diffusion for T1 brain tumor MRI.")
	subparsers = parser.add_subparsers(dest="mode")

	train_parser = subparsers.add_parser("train", help="Fine-tune a pretrained Stable Diffusion model.")
	train_parser.add_argument("--data-root", type=str, default="Dataset/T1")
	train_parser.add_argument("--output-dir", type=str, default="outputs/sd_t1_mri")
	train_parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
	train_parser.add_argument("--train-method", type=str, choices=["lora", "full"], default="lora")
	train_parser.add_argument("--lora-rank", type=int, default=8)
	train_parser.add_argument("--lora-alpha", type=int, default=16)
	train_parser.add_argument("--lora-dropout", type=float, default=0.0)
	train_parser.add_argument("--image-size", type=int, default=256)
	train_parser.add_argument("--batch-size", type=int, default=2)
	train_parser.add_argument("--epochs", type=int, default=5)
	train_parser.add_argument("--learning-rate", type=float, default=1e-4)
	train_parser.add_argument("--num-workers", type=int, default=2)
	train_parser.add_argument("--checkpoint-steps", type=int, default=500)
	train_parser.add_argument("--seed", type=int, default=42)
	train_parser.add_argument("--fp16", action="store_true")

	gen_parser = subparsers.add_parser("generate", help="Generate T1 MRI samples from a fine-tuned model.")
	gen_parser.add_argument("--model-path", type=str, default="outputs/sd_t1_mri")
	gen_parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
	gen_parser.add_argument("--lora-path", type=str, default="")
	gen_parser.add_argument("--output-dir", type=str, default="outputs/generated")
	gen_parser.add_argument(
		"--prompt",
		type=str,
		default="axial T1-weighted brain MRI scan showing glioma, medical imaging, grayscale",
	)
	gen_parser.add_argument("--negative-prompt", type=str, default="colorful, cartoon, low quality, distorted")
	gen_parser.add_argument("--num-images", type=int, default=8)
	gen_parser.add_argument("--inference-steps", type=int, default=40)
	gen_parser.add_argument("--guidance-scale", type=float, default=7.5)
	gen_parser.add_argument("--image-size", type=int, default=256)
	gen_parser.add_argument("--fp16", action="store_true")

	return parser


def main() -> None:
	parser = build_parser()
	if len(sys.argv) == 1:
		parser.print_help()
		print("\nExamples:")
		print(
			"  python model.py train --data-root Dataset/T1 --output-dir outputs/sd_t1_mri --base-model runwayml/stable-diffusion-v1-5 --epochs 5 --batch-size 2 --image-size 256 --fp16"
		)
		print(
			"  python model.py generate --model-path outputs/sd_t1_mri --output-dir outputs/generated --prompt \"axial T1-weighted brain MRI scan showing glioma, medical imaging, grayscale\" --num-images 8 --image-size 256 --fp16"
		)
		return

	args = parser.parse_args()

	if args.mode == "train":
		train(args)
	elif args.mode == "generate":
		generate(args)
	else:
		parser.error("Please choose a mode: 'train' or 'generate'.")


if __name__ == "__main__":
	main()
