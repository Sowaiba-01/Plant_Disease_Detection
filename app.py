import os
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

from src.predict import predict, load_model_and_classes, CONFIDENCE_THRESHOLD


# Load once at startup — not on every request
model, class_names = load_model_and_classes()


# ── Disease info for all 38 PlantVillage classes ──────────────────────────────
# FIX: old app.py only had 1 entry ("Healthy"). Every disease returned the same
# generic text, making the app useless. All 38 classes are now populated.
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "severity": "medium",
        "description": "Fungal disease causing dark, scaly lesions on leaves and fruit.",
        "treatment": "Apply fungicides (captan or myclobutanil). Remove fallen leaves. Prune for air circulation.",
    },
    "Apple___Black_rot": {
        "severity": "high",
        "description": "Fungal infection causing brown rotting of fruit and 'frog-eye' leaf spots.",
        "treatment": "Remove mummified fruit. Apply copper-based fungicide. Prune dead wood.",
    },
    "Apple___Cedar_apple_rust": {
        "severity": "medium",
        "description": "Fungal disease producing bright orange spots on leaves.",
        "treatment": "Apply myclobutanil fungicide early in season. Remove nearby cedar/juniper hosts.",
    },
    "Apple___healthy": {
        "severity": "none",
        "description": "No disease detected. Apple plant appears healthy.",
        "treatment": "Continue regular watering, fertilisation, and seasonal pruning.",
    },
    "Blueberry___healthy": {
        "severity": "none",
        "description": "No disease detected. Blueberry plant appears healthy.",
        "treatment": "Maintain soil pH 4.5–5.5. Regular irrigation and mulching.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "medium",
        "description": "Fungal disease creating white powdery coating on young leaves and shoots.",
        "treatment": "Apply sulfur-based or potassium bicarbonate fungicide. Improve air circulation.",
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "none",
        "description": "No disease detected. Cherry plant appears healthy.",
        "treatment": "Regular watering and balanced fertilisation. Monitor for pests.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "high",
        "description": "Fungal disease causing grey rectangular lesions on leaves, reducing photosynthesis.",
        "treatment": "Plant resistant hybrids. Apply strobilurin fungicide. Rotate crops.",
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "medium",
        "description": "Fungal disease producing small, cinnamon-brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicide (propiconazole) at early stages. Plant resistant varieties.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "high",
        "description": "Fungal disease causing long, cigar-shaped grey-green lesions on leaves.",
        "treatment": "Use resistant hybrids. Apply fungicide. Till crop residue after harvest.",
    },
    "Corn_(maize)___healthy": {
        "severity": "none",
        "description": "No disease detected. Corn plant appears healthy.",
        "treatment": "Ensure adequate nitrogen fertilisation and consistent irrigation.",
    },
    "Grape___Black_rot": {
        "severity": "high",
        "description": "Fungal disease causing brown leaf lesions and black shrivelled fruit (mummies).",
        "treatment": "Remove mummified fruit. Apply mancozeb or myclobutanil fungicide from budbreak.",
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "high",
        "description": "Complex fungal disease causing tiger-stripe leaf patterns and berry spotting.",
        "treatment": "Prune during dry weather. Apply wound protectants. Remove infected wood.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "medium",
        "description": "Fungal disease causing dark angular spots on leaves, leading to defoliation.",
        "treatment": "Apply copper-based fungicide. Remove infected leaves. Improve vineyard airflow.",
    },
    "Grape___healthy": {
        "severity": "none",
        "description": "No disease detected. Grapevine appears healthy.",
        "treatment": "Regular pruning, irrigation management, and seasonal fungicide programme.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "high",
        "description": "Devastating bacterial disease spread by psyllids. Causes blotchy yellowing and bitter, misshapen fruit. No cure exists.",
        "treatment": "Remove and destroy infected trees. Control Asian citrus psyllid with insecticides. Plant certified disease-free stock.",
    },
    "Peach___Bacterial_spot": {
        "severity": "medium",
        "description": "Bacterial infection causing small dark spots on leaves, fruit, and twigs.",
        "treatment": "Apply copper hydroxide sprays from bloom. Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Peach___healthy": {
        "severity": "none",
        "description": "No disease detected. Peach tree appears healthy.",
        "treatment": "Thin fruit for size and quality. Apply dormant copper spray annually.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "medium",
        "description": "Bacterial disease causing water-soaked spots on leaves and raised scab-like lesions on fruit.",
        "treatment": "Use copper-based bactericide. Avoid overhead watering. Rotate crops every 2–3 years.",
    },
    "Pepper,_bell___healthy": {
        "severity": "none",
        "description": "No disease detected. Bell pepper plant appears healthy.",
        "treatment": "Consistent watering, calcium fertilisation to prevent blossom end rot.",
    },
    "Potato___Early_blight": {
        "severity": "medium",
        "description": "Fungal disease causing dark concentric ring 'target' spots, starting on older leaves.",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. Remove infected foliage. Rotate crops.",
    },
    "Potato___Late_blight": {
        "severity": "high",
        "description": "Highly destructive oomycete disease (caused the Irish Famine). Causes water-soaked dark lesions that spread rapidly.",
        "treatment": "Apply systemic fungicide (metalaxyl) immediately. Remove infected plants. Avoid overhead irrigation.",
    },
    "Potato___healthy": {
        "severity": "none",
        "description": "No disease detected. Potato plant appears healthy.",
        "treatment": "Hilling, consistent moisture, and scout regularly for blight symptoms.",
    },
    "Raspberry___healthy": {
        "severity": "none",
        "description": "No disease detected. Raspberry plant appears healthy.",
        "treatment": "Prune old canes annually. Maintain good air circulation.",
    },
    "Soybean___healthy": {
        "severity": "none",
        "description": "No disease detected. Soybean plant appears healthy.",
        "treatment": "Monitor for soybean rust. Ensure proper plant spacing.",
    },
    "Squash___Powdery_mildew": {
        "severity": "medium",
        "description": "Fungal disease causing white powdery patches on leaf surfaces.",
        "treatment": "Apply potassium bicarbonate or neem oil. Remove heavily infected leaves.",
    },
    "Strawberry___Leaf_scorch": {
        "severity": "medium",
        "description": "Fungal disease causing small purple spots that enlarge into irregular brown lesions.",
        "treatment": "Apply captan fungicide. Remove old foliage after harvest. Avoid overhead watering.",
    },
    "Strawberry___healthy": {
        "severity": "none",
        "description": "No disease detected. Strawberry plant appears healthy.",
        "treatment": "Renovate beds after harvest. Maintain proper plant spacing.",
    },
    "Tomato___Bacterial_spot": {
        "severity": "medium",
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper bactericide. Use certified disease-free seed. Avoid working when plants are wet.",
    },
    "Tomato___Early_blight": {
        "severity": "medium",
        "description": "Fungal disease causing concentric ring lesions ('target spots') on lower leaves first.",
        "treatment": "Apply chlorothalonil fungicide. Remove lower infected leaves. Mulch to prevent soil splash.",
    },
    "Tomato___Late_blight": {
        "severity": "high",
        "description": "Rapidly spreading oomycete disease causing dark greasy lesions. Can destroy a crop in days.",
        "treatment": "Apply metalaxyl fungicide immediately. Remove and bag infected plants. Do not compost.",
    },
    "Tomato___Leaf_Mold": {
        "severity": "medium",
        "description": "Fungal disease causing yellow patches on upper leaf surface with olive-grey mould below.",
        "treatment": "Improve greenhouse ventilation. Apply copper or chlorothalonil fungicide.",
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "medium",
        "description": "Fungal disease causing circular spots with dark borders and light centres.",
        "treatment": "Apply mancozeb fungicide. Remove infected leaves. Stake plants to improve airflow.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "medium",
        "description": "Pest infestation causing stippled, bronze leaves with fine webbing on undersides.",
        "treatment": "Apply miticide or neem oil. Increase humidity. Introduce predatory mites.",
    },
    "Tomato___Target_Spot": {
        "severity": "medium",
        "description": "Fungal disease producing brown lesions with concentric rings and yellow halos.",
        "treatment": "Apply azoxystrobin fungicide. Remove heavily infected leaves.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "high",
        "description": "Viral disease spread by whiteflies causing severe leaf curling, yellowing, and stunted growth.",
        "treatment": "Control whitefly populations with insecticides. Remove infected plants. Use reflective mulch.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "high",
        "description": "Highly contagious virus causing mosaic yellowing pattern and distorted leaves.",
        "treatment": "Remove and destroy infected plants. Disinfect tools. Plant resistant varieties.",
    },
    "Tomato___healthy": {
        "severity": "none",
        "description": "No disease detected. Tomato plant appears healthy.",
        "treatment": "Regular staking, consistent watering, and calcium supplementation.",
    },
}

SEVERITY_EMOJI = {"none": "✅", "low": "🟡", "medium": "🟠", "high": "🔴"}


def get_disease_info(class_name: str) -> dict:
    """Look up disease info. Falls back gracefully for any unknown class."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    # Fuzzy fallback — handles minor name mismatches
    for key, info in DISEASE_INFO.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return info
    return {
        "severity": "medium",
        "description": f"Detected: {class_name.replace('_', ' ')}",
        "treatment": "Consult a local agronomist for targeted treatment advice.",
    }


def run_prediction(image: "Image.Image"):
    """
    Main prediction function called by Gradio.
    FIX: uses a unique temp file per request to avoid race conditions in
    multi-user deployments (old code wrote every image to /tmp/input_leaf.jpg).
    """
    if image is None:
        return {}, "⬅️  Please upload a leaf image to get started."

    # FIX: unique temp file per request — safe for concurrent users
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        result = predict(tmp_path, model=model, class_names=class_names)
    finally:
        os.unlink(tmp_path)   # always clean up

    # Low-confidence guard
    if result["low_confidence"]:
        warning = (
            f"⚠️ **Low confidence ({result['confidence']*100:.1f}%)**\n\n"
            f"The model is not confident about this image.\n"
            f"Please upload a **clear, well-lit photo of a single leaf**."
        )
        label_dict = {item["class"]: item["confidence"] for item in result["top3"]}
        return label_dict, warning

    # Build label dict for Gradio's gr.Label component
    label_dict = {item["class"]: item["confidence"] for item in result["top3"]}

    # Build markdown report using full disease info
    info           = get_disease_info(result["predicted_class"])
    confidence_pct = result["confidence"] * 100
    emoji          = SEVERITY_EMOJI.get(info.get("severity", "medium"), "🟠")
    display_name   = result["predicted_class"].replace("_", " ").replace("  ", " ")

    report = f"""## {emoji} {display_name}

**Confidence:** {confidence_pct:.1f}%  
**Severity:** {info.get('severity', 'unknown').capitalize()}

**About this disease:**  
{info['description']}

**Recommended treatment:**  
{info['treatment']}

---
*Top 3 predictions:*
"""
    for item in result["top3"]:
        bar = "█" * int(item["confidence"] * 20)
        report += f"\n- `{item['class'].replace('_', ' ')}` — {item['confidence']*100:.1f}%  {bar}"

    return label_dict, report


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Plant Disease Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌿 Plant Disease Detection
    Upload a clear photo of a **single plant leaf** to detect diseases using deep learning (EfficientNetB0 fine-tuned on PlantVillage).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload leaf image")
            predict_btn = gr.Button("🔍 Detect Disease", variant="primary")

        with gr.Column(scale=1):
            label_output  = gr.Label(num_top_classes=3, label="Top Predictions")
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

    gr.Markdown(f"""
    ---
    **Model:** EfficientNetB0 fine-tuned on PlantVillage &nbsp;|&nbsp;
    **Classes:** {len(class_names)} plant disease categories &nbsp;|&nbsp;
    **Confidence threshold:** {CONFIDENCE_THRESHOLD*100:.0f}% &nbsp;|&nbsp;
    [GitHub](https://github.com/Sowaiba-01/Plant_Disease_Detection)
    """)


if __name__ == "__main__":
    demo.launch(share=False)
