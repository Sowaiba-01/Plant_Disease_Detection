

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

from src.predict import predict, load_model_and_classes


# Load once at startup — not on every request
model, class_names = load_model_and_classes()

# Disease info: describe each class so the app is actually useful
DISEASE_INFO = {
    "Healthy": {
        "status": "Healthy",
        "description": "No disease detected. Your plant looks healthy!",
        "treatment": "Continue regular watering and fertilisation.",
        "severity": "none",
    },
}
# Fallback info for any class not in the dict above
DEFAULT_INFO = {
    "status": "Disease detected",
    "description": "A disease has been identified in this plant.",
    "treatment": "Consult a local agronomist for targeted treatment advice.",
    "severity": "medium",
}


def get_disease_info(class_name: str) -> dict:
    for key, info in DISEASE_INFO.items():
        if key.lower() in class_name.lower():
            return info
    return {**DEFAULT_INFO, "description": f"Detected: {class_name.replace('_', ' ')}"}


def run_prediction(image: Image.Image):
    """
    Main prediction function called by Gradio.
    Takes a PIL image, returns prediction label dict + markdown report.
    """
    if image is None:
        return {}, "Please upload a leaf image."

    # Save to temp file since predict() takes a path
    tmp_path = "/tmp/input_leaf.jpg"
    image.save(tmp_path)

    result = predict(tmp_path, model=model, class_names=class_names)

    # Build label dict for Gradio's gr.Label component
    label_dict = {item["class"]: item["confidence"] for item in result["top3"]}

    # Build markdown report
    info = get_disease_info(result["predicted_class"])
    confidence_pct = result["confidence"] * 100
    severity_emoji = {"none": "✅", "low": "🟡", "medium": "🟠", "high": "🔴"}.get(
        info.get("severity", "medium"), "🟠"
    )

    report = f"""
## {severity_emoji} {result['predicted_class'].replace('_', ' ')}

**Confidence:** {confidence_pct:.1f}%

**Description:** {info['description']}

**Recommended action:** {info['treatment']}

---
*Top predictions:*
"""
    for item in result["top3"]:
        bar_len = int(item["confidence"] * 20)
        report += f"\n- `{item['class']}` — {item['confidence']*100:.1f}%"

    return label_dict, report


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Plant Disease Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌿 Plant Disease Detection
    Upload a photo of a plant leaf to detect diseases using deep learning (EfficientNetB0).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload leaf image")
            predict_btn = gr.Button("Detect Disease", variant="primary")
            gr.Examples(
                examples=[],  # Add example images here after training
                inputs=image_input,
            )

        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=3,
                label="Prediction"
            )
            report_output = gr.Markdown(label="Analysis Report")

    predict_btn.click(
        fn=run_prediction,
        inputs=image_input,
        outputs=[label_output, report_output],
    )

    image_input.change(
        fn=run_prediction,
        inputs=image_input,
        outputs=[label_output, report_output],
    )

    gr.Markdown("""
    ---
    **Model:** EfficientNetB0 fine-tuned on PlantVillage dataset |
    **Classes:** 38 plant disease categories |
    [GitHub](https://github.com/Sowaiba-01/Plant_Disease_Detection)
    """)


if __name__ == "__main__":
    demo.launch(share=False)
