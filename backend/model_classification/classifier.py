import torch
import torch.nn as nn
import torchvision.models as models


class TumorClassifier(nn.Module):
    """
    Class description:
        Binary brain tumor classification model backed by ResNet18.

    Attributes:
        model (nn.Module): ResNet18 model with a replacement classification head.
    """

    def __init__(self, num_classes=2, pretrained=True):
        """
        Function description:
            Initialize the ResNet18 classifier architecture.

        Parameters:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to start from ImageNet pretrained weights.

        Returns:
            None
        """
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
        """
        Function description:
            Run a forward pass through the classifier.

        Parameters:
            x (torch.Tensor): Input image batch tensor.

        Returns:
            torch.Tensor: Raw class logits.
        """
        return self.model(x)


def build_classifier(device=None):
    """
    Function description:
        Build and return the classifier model.

    Parameters:
        device (torch.device | None): Optional device where the model should be placed.

    Returns:
        TumorClassifier: Classifier model on the selected device.
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
    Function description:
        Load a trained classifier checkpoint.

    Parameters:
        checkpoint_path (str | Path): Path to the saved checkpoint.
        device (torch.device | None): Optional device used for loading and inference.

    Returns:
        TumorClassifier: Classifier model loaded with checkpoint weights.
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

    # Fix old checkpoints saved without "model." and normalize fc naming.
    # Older classifier checkpoints may have used a Sequential wrapper for the
    # final block, storing the linear weights under "fc.1" instead of "fc".
    new_state_dict = {}

    for key, value in state_dict.items():

        if not key.startswith("model."):
            new_key = f"model.{key}"
        else:
            new_key = key

        if new_key.startswith("model.fc.1."):
            new_key = new_key.replace("model.fc.1.", "model.fc.")

        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    model.eval()

    print(f"Loaded classifier from: {checkpoint_path}")

    return model


def get_loss_function():
    """
    Function description:
        Create the loss function used for binary classification training.

    Parameters:
        None

    Returns:
        nn.CrossEntropyLoss: Cross-entropy loss function.
    """

    return nn.CrossEntropyLoss()


def get_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=1e-5
):
    """
    Function description:
        Create an AdamW optimizer for classifier training.

    Parameters:
        model (nn.Module): Model whose parameters should be optimized.
        learning_rate (float): Optimizer learning rate.
        weight_decay (float): AdamW weight decay value.

    Returns:
        torch.optim.AdamW: Optimizer configured for the model.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer
