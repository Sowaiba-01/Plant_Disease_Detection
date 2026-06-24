"""
Prediction utility for plant disease detection.
Replaces main6.py — no hardcoded paths, no easygui dependency.

Usage:
    python src/predict.py --image path/to/leaf.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)
MODEL_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")


def load_model_and_classes():
    """Load model and class names. Raises clear errors if files are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            f"Run: python src/train.py  to train the model first."
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found at {CLASS_NAMES_PATH}.\n"
            f"Run: python src/train.py  to generate them."
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict(image_path: str, model=None, class_names=None) -> dict:
    """
    Run inference on a single image.

    Returns:
        dict with keys: predicted_class, confidence, all_scores
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]  # Shape: (num_classes,)

    top_idx = int(np.argmax(scores))
    confidence = float(scores[top_idx])

    # Top-3 predictions
    top3_indices = np.argsort(scores)[::-1][:3]
    top3 = [
        {"class": class_names[i], "confidence": float(scores[i])}
        for i in top3_indices
    ]

    return {
        "predicted_class": class_names[top_idx],
        "confidence": confidence,
        "top3": top3,
        "all_scores": {class_names[i]: float(scores[i]) for i in range(len(class_names))}
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

    print(f"\nPrediction:  {result['predicted_class']}")
    print(f"Confidence:  {result['confidence']*100:.1f}%")
    print("\nTop 3 predictions:")
    for i, item in enumerate(result["top3"], 1):
        bar = "█" * int(item["confidence"] * 20)
        print(f"  {i}. {item['class']:<35} {item['confidence']*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
