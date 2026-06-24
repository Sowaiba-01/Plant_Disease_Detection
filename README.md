# Plant Disease Detection

A deep learning system that detects and classifies **38 plant disease categories** from leaf images, built with EfficientNetB0 transfer learning and deployed via a Gradio web interface.

---

## Results

| Metric | Value |
|---|---|
| Model | EfficientNetB0 (fine-tuned) |
| Dataset | PlantVillage (54,305 images) |
| Classes | 38 (14 crop species) |
| Expected val accuracy | 93–97% |
| Input size | 224 × 224 RGB |
| Confidence threshold | 50% (low-confidence inputs rejected) |

---


## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download [PlantVillage from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract to `data/PlantVillage/`.

### 3. Train the model
```bash
export DATA_DIR="data/PlantVillage"   # Linux/Mac
set DATA_DIR=data/PlantVillage        # Windows

python -m src.train
```
This runs two training phases:
- **Phase 1** (15 epochs): trains only the classification head (backbone frozen)
- **Phase 2** (10 epochs): fine-tunes top 20 layers of EfficientNetB0 at low LR

Model is saved to `models/best_model.keras`.

### 4. Run the web app
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

### 5. Run inference from CLI
```bash
python -m src.predict --image path/to/leaf.jpg
```

---

## Running Tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```
Tests use mocked models — no GPU or downloaded weights required.

---

## Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — model training and inference
- **EfficientNetB0** — pretrained backbone (ImageNet weights)
- **Gradio** — web interface
- **Pillow / NumPy** — image processing
- **pytest** — testing

---

## Dataset

[PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — 54,305 labeled leaf images across 38 classes (14 crop species).

---

## License

MIT License — see [LICENSE](LICENSE).
