

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE         = (224, 224)
MODEL_PATH       = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
CONFIDENCE_THRESHOLD = 0.50   # FIX: reject predictions below this — old code had no threshold


def load_model_and_classes(
    model_path: Path = MODEL_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
):
    """Load model and class names. Raises clear errors if files are missing."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            f"Run:  python -m src.train   to train the model first."
        )
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names not found at {class_names_path}.\n"
            f"Run:  python -m src.train   to generate them."
        )

    model = tf.keras.models.load_model(str(model_path))
    with open(class_names_path) as f:
        class_names = json.load(f)

    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for EfficientNetB0 inference.
    EfficientNet expects pixel values in [0, 255] — no division by 255.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)          # keep [0, 255] range
    img_array = np.expand_dims(img_array, axis=0)        # add batch dimension → (1, 224, 224, 3)
    return img_array


def predict(image_path: str, model=None, class_names=None) -> dict:
    """
    Run inference on a single image.

    Returns dict with keys:
        predicted_class  – top predicted class name (or "Unknown" if low confidence)
        confidence       – float in [0, 1]
        top3             – list of {class, confidence} dicts, sorted descending
        all_scores       – dict of all class probabilities
        low_confidence   – True if confidence < CONFIDENCE_THRESHOLD
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    img_array   = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    scores      = predictions[0]                         # shape: (num_classes,)

    top_idx    = int(np.argmax(scores))
    confidence = float(scores[top_idx])

    # Top-3 predictions sorted highest → lowest
    top3_indices = np.argsort(scores)[::-1][:3]
    top3 = [
        {"class": class_names[i], "confidence": float(scores[i])}
        for i in top3_indices
    ]

    # FIX: confidence threshold — flag low-confidence predictions instead of
    # silently returning a wrong answer (e.g. when user uploads a non-leaf image)
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    predicted_class = class_names[top_idx] if not low_confidence else "Unknown"

    return {
        "predicted_class": predicted_class,
        "confidence":      confidence,
        "top3":            top3,
        "all_scores":      {class_names[i]: float(scores[i]) for i in range(len(class_names))},
        "low_confidence":  low_confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detector")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return

    print(f"Analyzing: {args.image}")
    result = predict(args.image)

    if result["low_confidence"]:
        print(f"\n⚠  Low confidence ({result['confidence']*100:.1f}%) — please upload a clearer leaf image.")
    else:
        print(f"\nPrediction : {result['predicted_class']}")
        print(f"Confidence : {result['confidence']*100:.1f}%")

    print("\nTop 3 predictions:")
    for i, item in enumerate(result["top3"], 1):
        bar = "█" * int(item["confidence"] * 20)
        print(f"  {i}. {item['class']:<40} {item['confidence']*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
