# preprocessing.py

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# ==========================================
# Image Size
# ==========================================

IMAGE_SIZE = 224


# ==========================================
# Training Transforms
# ==========================================

train_transforms = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ==========================================
# Validation / Test Transforms
# ==========================================

val_test_transforms = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ==========================================
# Dataset Class
# ==========================================

class BrainTumorDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels=None,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return with label if exists
        if self.labels is not None:

            label = self.labels[idx]

            return image, label

        return image


# ==========================================
# Preprocess Single Image
# ==========================================

def preprocess_single_image(
    image,
    transform=val_test_transforms
):
    """
    Preprocess single image for inference

    Input:
        PIL image or numpy array

    Output:
        Tensor shape:
        [1, 3, 224, 224]
    """

    # Convert numpy -> PIL
    if isinstance(image, np.ndarray):

        image = Image.fromarray(image)

    # Ensure RGB
    image = image.convert("RGB")

    # Apply transform
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ==========================================
# Class Label Mapping
# ==========================================

CLASS_NAMES = {
    0: "No Tumor",
    1: "Tumor"
}