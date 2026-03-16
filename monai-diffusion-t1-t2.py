import os
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    parser.add_argument("--train_split", type=float, default=0.8)

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

    # make runs more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def split_data(data_dicts, train_split=0.8, seed=42):
    shuffled = data_dicts.copy()
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) < 2:
        raise ValueError("Need at least 2 paired cases to create train/validation split.")

    split_idx = max(1, int(len(shuffled) * train_split))
    split_idx = min(split_idx, len(shuffled) - 1)

    train_files = shuffled[:split_idx]
    val_files = shuffled[split_idx:]

    if len(val_files) == 0:
        raise ValueError("Validation set is empty. Need at least 2 paired cases.")

    return train_files, val_files


def get_transforms(args, train=True):
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
    ]

    if train:
        transforms.extend(
            [
                RandSpatialCropd(
                    keys=["t1", "t2"],
                    roi_size=roi_size,
                    random_size=False,
                ),
                RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["t1", "t2"], prob=0.5, max_k=3),
            ]
        )
    else:
        transforms.append(
            RandSpatialCropd(
                keys=["t1", "t2"],
                roi_size=roi_size,
                random_size=False,
            )
        )

    transforms.append(EnsureTyped(keys=["t1", "t2"]))
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


def evaluate_loss(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in loader:
            t1 = batch_data["t1"].to(device)
            t2 = batch_data["t2"].to(device)

            pred_t2 = model(t1)
            loss = loss_function(pred_t2, t2)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return float("inf")

    return total_loss / num_batches


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, config, history):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
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
    split_path = os.path.join(output_dir, "data_split.json")

    device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")

    data_dicts = build_data_dicts(t1_dir, t2_dir)
    train_files, val_files = split_data(
        data_dicts,
        train_split=args.train_split,
        seed=args.seed,
    )

    print(f"Total paired cases: {len(data_dicts)}")
    print(f"Train cases: {len(train_files)}")
    print(f"Val cases: {len(val_files)}")

    save_json(
        {
            "train": train_files,
            "val": val_files,
        },
        split_path,
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=get_transforms(args, train=True),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=get_transforms(args, train=False),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    first_batch = next(iter(train_loader))
    print("T1 shape:", tuple(first_batch["t1"].shape))
    print("T2 shape:", tuple(first_batch["t2"].shape))

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.L1Loss()

    config = {
        "dataset_root": dataset_root,
        "t1_dir": t1_dir,
        "t2_dir": t2_dir,
        "output_dir": output_dir,
        "seed": args.seed,
        "train_split": args.train_split,
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
    }

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            t1 = batch_data["t1"].to(device)
            t2 = batch_data["t2"].to(device)

            optimizer.zero_grad()
            pred_t2 = model(t1)
            loss = loss_function(pred_t2, t2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)
        avg_val_loss = evaluate_loss(model, val_loader, loss_function, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch}/{args.max_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        latest_path = os.path.join(model_dir, "latest_model.pt")
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            config,
            history,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(model_dir, "best_model.pt")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
                history,
            )
            print(f"Best model updated and saved to: {best_path}")

        save_json(history, history_path)

    print("\nTraining finished.")
    print(f"Latest model: {os.path.join(model_dir, 'latest_model.pt')}")
    print(f"Best model:   {os.path.join(model_dir, 'best_model.pt')}")
    print(f"History:      {history_path}")
    print(f"Split file:   {split_path}")


if __name__ == "__main__":
    main()