import argparse
import json
from pathlib import Path


def load_history(history_path: Path) -> dict:
    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_history(history: dict) -> dict:
    epoch_metrics = history.get("epoch_metrics", [])
    if not epoch_metrics:
        raise ValueError("No epoch metrics found in training history.")

    train_losses = [x["train_loss"] for x in epoch_metrics]
    val_losses = [x["val_loss"] for x in epoch_metrics]

    best_epoch_record = min(epoch_metrics, key=lambda x: x["val_loss"])
    final_epoch_record = epoch_metrics[-1]

    overfitting_flag = final_epoch_record["val_loss"] > best_epoch_record["val_loss"]

    summary = {
        "train_method": history.get("train_method"),
        "base_model": history.get("base_model"),
        "epochs": history.get("epochs"),
        "batch_size": history.get("batch_size"),
        "image_size": history.get("image_size"),
        "learning_rate": history.get("learning_rate"),
        "trainable_parameters": history.get("trainable_parameters"),
        "train_size": history.get("train_size"),
        "val_size": history.get("val_size"),
        "total_training_time_sec": history.get("total_training_time_sec"),
        "initial_train_loss": train_losses[0],
        "final_train_loss": train_losses[-1],
        "initial_val_loss": val_losses[0],
        "final_val_loss": val_losses[-1],
        "best_epoch": best_epoch_record["epoch"],
        "best_val_loss": best_epoch_record["val_loss"],
        "best_train_loss_at_best_epoch": best_epoch_record["train_loss"],
        "overfitting_flag": overfitting_flag,
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Model Performance Summary ===")
    print(f"Training method         : {summary['train_method']}")
    print(f"Base model              : {summary['base_model']}")
    print(f"Epochs                  : {summary['epochs']}")
    print(f"Batch size              : {summary['batch_size']}")
    print(f"Image size              : {summary['image_size']}")
    print(f"Learning rate           : {summary['learning_rate']}")
    print(f"Trainable parameters    : {summary['trainable_parameters']:,}")
    print(f"Train samples           : {summary['train_size']}")
    print(f"Validation samples      : {summary['val_size']}")
    print(f"Training time (sec)     : {summary['total_training_time_sec']:.2f}")
    print(f"Initial train loss      : {summary['initial_train_loss']:.6f}")
    print(f"Final train loss        : {summary['final_train_loss']:.6f}")
    print(f"Initial val loss        : {summary['initial_val_loss']:.6f}")
    print(f"Final val loss          : {summary['final_val_loss']:.6f}")
    print(f"Best epoch              : {summary['best_epoch']}")
    print(f"Best validation loss    : {summary['best_val_loss']:.6f}")
    print(f"Overfitting detected    : {summary['overfitting_flag']}")


def save_summary(summary: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diffusion model training performance.")
    parser.add_argument(
        "--history-path",
        type=str,
        default="outputs/sd_t1_mri/training_history.json",
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/sd_t1_mri/model_performance_summary.json",
        help="Path to save summary JSON",
    )
    args = parser.parse_args()

    history_path = Path(args.history_path)
    output_path = Path(args.output_path)

    history = load_history(history_path)
    summary = summarize_history(history)
    print_summary(summary)
    save_summary(summary, output_path)


if __name__ == "__main__":
    main()