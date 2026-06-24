"""
Train the plant disease detection model.

Usage:
    export DATA_DIR="data/PlantVillage"
    python src/train.py

Or on Windows:
    set DATA_DIR=data/PlantVillage
    python src/train.py
"""

import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model import build_model, unfreeze_top_layers


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.getenv("DATA_DIR", "data/PlantVillage")
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.keras"
HISTORY_PATH = MODEL_DIR / "history.json"
PLOTS_DIR  = Path("reports/figures")

EPOCHS_HEAD  = 15   # Phase 1: train only the classification head
EPOCHS_FINETUNE = 10  # Phase 2: fine-tune top layers of backbone
BATCH_SIZE   = 32
# ──────────────────────────────────────────────────────────────────────────────


def get_callbacks(model_path: Path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=1
        ),
    ]


def plot_training_history(history_dict: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_dict["accuracy"], label="Train accuracy", marker="o")
    axes[0].plot(history_dict["val_accuracy"], label="Val accuracy", marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_dict["loss"], label="Train loss", marker="o")
    axes[1].plot(history_dict["val_loss"], label="Val loss", marker="o")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {out}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_ds, val_ds, class_names = load_datasets(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Save class names so predict.py and app.py can load them
    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # ── Phase 1: Train head only ───────────────────────────────────────────
    print("\nPhase 1 — Training classification head (backbone frozen)...")
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(MODEL_PATH),
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────
    print("\nPhase 2 — Fine-tuning top layers of EfficientNetB0...")
    model = unfreeze_top_layers(model, num_layers=20)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR!
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=get_callbacks(MODEL_PATH),
    )

    # Merge histories for plotting
    combined = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        combined[key] = history1.history[key] + history2.history[key]

    with open(HISTORY_PATH, "w") as f:
        json.dump(combined, f)

    plot_training_history(combined, PLOTS_DIR)

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"\nFinal val accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Final val loss:     {loss:.4f}")
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
