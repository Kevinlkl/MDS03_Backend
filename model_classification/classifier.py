import torch
import torch.nn as nn
import torchvision.models as models


class TumorClassifier(nn.Module):
    """
    Binary Brain Tumor Classification Model
    ---------------------------------------
    Class 0 = No Tumor
    Class 1 = Tumor
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(TumorClassifier, self).__init__()

        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            self.model = models.resnet18(weights=None)

        # Replace final fully connected layer
        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


def build_classifier(device=None):
    """
    Build and return classifier model
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"Using device: {device}")

    model = TumorClassifier(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    return model


def load_classifier_weights(
    checkpoint_path,
    device=None
):
    """
    Load trained classifier checkpoint
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Build model
    model = build_classifier(device)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Handle both:
    # 1. pure state_dict
    # 2. checkpoint dictionary
    # Handle checkpoint loading
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    # Fix old checkpoints saved without "model."
    new_state_dict = {}

    for key, value in state_dict.items():

        if not key.startswith("model."):
            new_key = f"model.{key}"
        else:
            new_key = key

        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    model.eval()

    print(f"Loaded classifier from: {checkpoint_path}")

    return model


def get_loss_function():
    """
    Cross entropy loss for binary classification
    """

    return nn.CrossEntropyLoss()


def get_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=1e-5
):
    """
    AdamW optimizer
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer