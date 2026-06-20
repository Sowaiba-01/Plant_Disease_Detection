# Plant Disease Detection

A deep learning system that detects and classifies plant diseases from leaf images using a Convolutional Neural Network (CNN).

## Overview

Plant diseases are a major threat to global food security. Early detection helps farmers act before crops are lost. This project uses a CNN trained on the PlantVillage dataset to classify leaf images into 8 categories — identifying whether a plant is healthy or affected by a specific disease.


## How It Works

1. **Training (`training.py`)** — Loads labeled leaf images, builds a CNN with 3 convolutional layers, and trains it to classify images into 8 disease/health categories.
2. **Prediction (`prediction.py`)** — Loads the trained model, lets you pick an image via a file dialog, and predicts the disease class with a confidence score.

## Tech Stack

- Python
- TensorFlow / Keras
- EasyGUI (file selection)
- NumPy, Matplotlib

## Dataset

Trained on the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — a public dataset of labeled plant leaf images.


## Notes

- Update the `data_dir` and `model_path` variables in `training.py` and `prediction.py` to match your local dataset/model location before running.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
