import os
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


def build_data_dicts(root_dir: str):
    root = Path(root_dir)
    data_dicts = []

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue

        t1_path = patient_dir / "t1.nii.gz"
        t2_path = patient_dir / "t2.nii.gz"

        if t1_path.exists() and t2_path.exists():
            data_dicts.append({
                "t1": str(t1_path),
                "t2": str(t2_path),
            })

    if not data_dicts:
        raise ValueError("No paired T1/T2 volumes found.")

    return data_dicts


def get_transforms(train=True):
    transforms = [
        LoadImaged(keys=["t1", "t2"]),
        EnsureChannelFirstd(keys=["t1", "t2"]),
        Orientationd(keys=["t1", "t2"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t2"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["t1", "t2"],
            a_min=0,
            a_max=2000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["t1", "t2"], source_key="t1"),
    ]

    if train:
        transforms.extend([
            RandSpatialCropd(keys=["t1", "t2"], roi_size=(96, 96, 96), random_size=False),
            RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["t1", "t2"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["t1", "t2"], prob=0.5, max_k=3),
        ])
    else:
        transforms.append(
            RandSpatialCropd(keys=["t1", "t2"], roi_size=(96, 96, 96), random_size=False)
        )

    transforms.append(EnsureTyped(keys=["t1", "t2"]))
    return Compose(transforms)


def main():
    root_dir = "dataset"
    data_dicts = build_data_dicts(root_dir)

    split_idx = max(1, int(0.8 * len(data_dicts)))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]

    train_ds = CacheDataset(
        data=train_files,
        transform=get_transforms(train=True),
        cache_rate=1.0,
        num_workers=2,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=get_transforms(train=False),
        cache_rate=1.0,
        num_workers=2,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet(
        spatial_dims=3,
        in_channels=1,   # input is T1
        out_channels=1,  # output is T2
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.L1Loss()  # good simple baseline for translation

    # shape check
    batch = next(iter(train_loader))
    print("T1 shape:", batch["t1"].shape)
    print("T2 shape:", batch["t2"].shape)

    max_epochs = 10

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch_data in enumerate(train_loader):
            t1 = batch_data["t1"].to(device)
            t2 = batch_data["t2"].to(device)

            optimizer.zero_grad()
            pred_t2 = model(t1)
            loss = loss_function(pred_t2, t2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / (step + 1)
        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}")

        # simple validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vstep, batch_data in enumerate(val_loader):
                t1 = batch_data["t1"].to(device)
                t2 = batch_data["t2"].to(device)

                pred_t2 = model(t1)
                loss = loss_function(pred_t2, t2)
                val_loss += loss.item()

        avg_val_loss = val_loss / (vstep + 1)
        print(f"Epoch {epoch + 1}/{max_epochs}, Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    main()