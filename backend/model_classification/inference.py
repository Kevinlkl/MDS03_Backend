import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from model_classification.classifier import load_classifier_weights
from model_classification.preprocessing import preprocess_single_image, CLASS_NAMES


def predict_single_image(
    image,
    checkpoint_path,
    device=None
):
    """
    Run inference on one image.
    Used for frontend demo upload.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = load_classifier_weights(
        checkpoint_path=checkpoint_path,
        device=device
    )

    if isinstance(image, (list, tuple)):
        if len(image) == 0:
            raise ValueError("No images provided for inference.")

        best_probs = None
        max_tumor_prob = 0

        for pil_img in image:
            image_tensor = preprocess_single_image(pil_img)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)[0]

                no_tumor_prob = probs[0].item()
                tumor_prob = probs[1].item()

                if tumor_prob > max_tumor_prob:
                    max_tumor_prob = tumor_prob
                    best_probs = {
                        "prob_no_tumor": no_tumor_prob,
                        "prob_tumor": tumor_prob
                    }

        prediction = (
            "Tumor"
            if best_probs["prob_tumor"] > 0.5
            else "No Tumor"
        )

        return {
            "prediction": prediction,
            "confidence": round(best_probs["prob_tumor" if prediction == "Tumor" else "prob_no_tumor"], 4),
            "prob_no_tumor": round(best_probs["prob_no_tumor"], 4),
            "prob_tumor": round(best_probs["prob_tumor"], 4)
        }

    image_tensor = preprocess_single_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "prob_no_tumor": round(probs[0, 0].item(), 4),
        "prob_tumor": round(probs[0, 1].item(), 4)
    }


def evaluate_classifier(
    checkpoint_path,
    test_loader,
    device=None
):
    """
    Evaluate classifier on a test dataset.
    Used for precomputed benchmark metrics.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = load_classifier_weights(
        checkpoint_path=checkpoint_path,
        device=device
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):

            if isinstance(batch, dict):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )
    test_recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )
    test_f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        test_auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=["No Tumor", "Tumor"],
        zero_division=0,
        output_dict=True
    )

    return {
        "accuracy": round(test_acc, 4),
        "precision": round(test_precision, 4),
        "recall": round(test_recall, 4),
        "f1_score": round(test_f1, 4),
        "auc": round(test_auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def compare_two_classifiers(
    image,
    baseline_checkpoint_path,
    synthetic_aug_checkpoint_path,
    device=None
):
    """
    Run one uploaded image through:
    1. baseline classifier
    2. synthetic-augmented classifier
    """

    baseline_result = predict_single_image(
        image=image,
        checkpoint_path=baseline_checkpoint_path,
        device=device
    )

    synthetic_aug_result = predict_single_image(
        image=image,
        checkpoint_path=synthetic_aug_checkpoint_path,
        device=device
    )

    return {
        "baseline_classifier": baseline_result,
        "synthetic_augmented_classifier": synthetic_aug_result
    }