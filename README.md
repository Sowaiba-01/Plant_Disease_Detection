# Plant Disease Detection

![Tests](https://github.com/Sowaiba-01/Plant_Disease_Detection/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A deep learning system that detects and classifies **38 plant disease categories** from leaf images. Built with EfficientNetB0 transfer learning and deployed as a Gradio web app.

Upload a photo of a plant leaf → get an instant diagnosis with disease description, severity, and treatment advice.

---


## Results

| | |
|---|---|
| **Model** | EfficientNetB0 (fine-tuned, two-phase) |
| **Dataset** | PlantVillage — 54,305 images |
| **Classes** | 38 across 14 crop species |
| **Expected accuracy** | 93–97% on validation set |
| **Input size** | 224 × 224 RGB |
| **Confidence threshold** | 50% — low-confidence predictions rejected |

### Supported crops & diseases

Apple (scab, black rot, cedar rust, healthy) · Blueberry · Cherry (powdery mildew) · Corn (cercospora, common rust, northern leaf blight) · Grape (black rot, esca, leaf blight) · Orange (citrus greening) · Peach (bacterial spot) · Bell Pepper (bacterial spot) · Potato (early blight, late blight) · Raspberry · Soybean · Squash (powdery mildew) · Strawberry (leaf scorch) · Tomato (bacterial spot, early blight, late blight, leaf mold, septoria, spider mites, target spot, yellow leaf curl virus, mosaic virus)

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Sowaiba-01/Plant_Disease_Detection.git
cd Plant_Disease_Detection
pip install -r requirements.txt
```

### 2. Download the dataset

Download [PlantVillage from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract so the structure is:

```
data/
└── PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    ├── Tomato___Late_blight/
    └── ...
```

### 3. Train

```bash
# Linux / Mac
export DATA_DIR="data/PlantVillage"
python -m src.train

# Windows
set DATA_DIR=data/PlantVillage
python -m src.train
```

Training runs in two phases automatically:
- **Phase 1** (up to 15 epochs): classification head only, backbone frozen, LR = 1e-3
- **Phase 2** (up to 10 epochs): top 20 EfficientNetB0 layers unfrozen, LR = 1e-5

Saves `models/best_model.keras` and `models/class_names.json`.

### 4. Run the web app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### 5. Run CLI prediction

```bash
python -m src.predict --image path/to/leaf.jpg
```

---

## Running Tests

Tests use mocked models — no GPU, no trained model, no internet required.

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Tech Stack

| | |
|---|---|
| **Model** | EfficientNetB0 via TensorFlow/Keras |
| **UI** | Gradio |
| **Image processing** | Pillow, NumPy |
| **Testing** | pytest, pytest-cov |
| **CI** | GitHub Actions |

---

## License

MIT — see [LICENSE](LICENSE).
