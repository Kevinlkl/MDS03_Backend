import os
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.networks.nets import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train MONAI 3D UNet for T1 -> T2 synthesis")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root containing T1/ and T2/ folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/monai3d-t1-to-t2",
        help="Directory to save models and training history",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_rate", type=float, default=1.0)

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)

    parser.add_argument("--pixdim_x", type=float, default=1.5)
    parser.add_argument("--pixdim_y", type=float, default=1.5)
    parser.add_argument("--pixdim_z", type=float, default=1.5)

    parser.add_argument("--intensity_a_min", type=float, default=0.0)
    parser.add_argument("--intensity_a_max", type=float, default=2000.0)
    parser.add_argument("--intensity_b_min", type=float, default=0.0)
    parser.add_argument("--intensity_b_max", type=float, default=1.0)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Warning: Training on CPU. This will be very slow.")
    return device


def build_data_dicts(t1_dir: str, t2_dir: str):
    """
    Pair T1 and T2 scans by identical filename.

    Example:
        T1/patient001.nii.gz
        T2/patient001.nii.gz
    """
    t1_dir = Path(t1_dir)
    t2_dir = Path(t2_dir)

    if not t1_dir.exists():
        raise FileNotFoundError(f"T1 folder not found: {t1_dir}")
    if not t2_dir.exists():
        raise FileNotFoundError(f"T2 folder not found: {t2_dir}")

    data_dicts = []
    missing_t2 = []

    for t1_file in sorted(t1_dir.glob("*.nii*")):
        t2_file = t2_dir / t1_file.name
        if t2_file.exists():
            data_dicts.append(
                {
                    "patient_id": t1_file.stem.replace(".nii", ""),
                    "t1": str(t1_file),
                    "t2": str(t2_file),
                }
            )
        else:
            missing_t2.append(t1_file.name)

    if missing_t2:
        print("\nWarning: the following T1 files have no matching T2:")
        for name in missing_t2:
            print(f"  - {name}")

    if not data_dicts:
        raise ValueError("No paired T1/T2 volumes found.")

    return data_dicts


def get_transforms(args):
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    pixdim = (args.pixdim_x, args.pixdim_y, args.pixdim_z)

    transforms = [
        LoadImaged(keys=["t1", "t2"]),
        EnsureChannelFirstd(keys=["t1", "t2"]),
        Orientationd(keys=["t1", "t2"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t2"],
            pixdim=pixdim,
            mode=("bilinear", "bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["t1", "t2"],
            a_min=args.intensity_a_min,
            a_max=args.intensity_a_max,
            b_min=args.intensity_b_min,
            b_max=args.intensity_b_max,
            clip=True,
        ),
        CropForegroundd(keys=["t1", "t2"], source_key="t1"),
        RandSpatialCropd(
            keys=["t1", "t2"],
            roi_size=roi_size,
            random_size=False,
        ),
        RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["t1", "t2"], prob=0.5, max_k=3),
        EnsureTyped(keys=["t1", "t2"]),
    ]

    return Compose(transforms)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    return model


def save_checkpoint(path, model, optimizer, epoch, config, history):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = args.dataset_root
    t1_dir = os.path.join(dataset_root, "T1")
    t2_dir = os.path.join(dataset_root, "T2")

    output_dir = args.output_dir
    model_dir = os.path.join(output_dir, "models")
    history_path = os.path.join(output_dir, "training_history.json")
    paired_data_path = os.path.join(output_dir, "paired_data.json")

    device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")

    data_dicts = build_data_dicts(t1_dir, t2_dir)
    print(f"Total paired cases used for full training: {len(data_dicts)}")

    save_json(
        {
            "all_training_data": data_dicts,
        },
        paired_data_path,
    )

    train_ds = CacheDataset(
        data=data_dicts,
        transform=get_transforms(args),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    first_batch = next(iter(train_loader))
    print("T1 shape:", tuple(first_batch["t1"].shape))
    print("T2 shape:", tuple(first_batch["t2"].shape))

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.L1Loss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    config = {
        "dataset_root": dataset_root,
        "t1_dir": t1_dir,
        "t2_dir": t2_dir,
        "output_dir": output_dir,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cache_rate": args.cache_rate,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "roi_size": [args.roi_x, args.roi_y, args.roi_z],
        "pixdim": [args.pixdim_x, args.pixdim_y, args.pixdim_z],
        "intensity_a_min": args.intensity_a_min,
        "intensity_a_max": args.intensity_a_max,
        "intensity_b_min": args.intensity_b_min,
        "intensity_b_max": args.intensity_b_max,
        "model_name": "UNet3D_T1_to_T2",
        "device": str(device),
        "training_mode": "full_training_no_validation",
        "amp_enabled": (device.type == "cuda"),
    }

    history = {
        "train_loss": [],
    }

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            t1 = batch_data["t1"].to(device, non_blocking=(device.type == "cuda"))
            t2 = batch_data["t2"].to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                pred_t2 = model(t1)
                loss = loss_function(pred_t2, t2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)

        print(f"Epoch {epoch}/{args.max_epochs} | Train Loss: {avg_train_loss:.6f}")

        latest_path = os.path.join(model_dir, "latest_model.pt")
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            config,
            history,
        )

        epoch_path = os.path.join(model_dir, f"model_epoch_{epoch}.pt")
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            epoch,
            config,
            history,
        )

        save_json(history, history_path)

    final_path = os.path.join(model_dir, "final_model.pt")
    save_checkpoint(
        final_path,
        model,
        optimizer,
        args.max_epochs,
        config,
        history,
    )

    print("\nTraining finished.")
    print(f"Latest model: {os.path.join(model_dir, 'latest_model.pt')}")
    print(f"Final model:  {os.path.join(model_dir, 'final_model.pt')}")
    print(f"History:      {history_path}")
    print(f"Data list:    {paired_data_path}")


if __name__ == "__main__":
    main()