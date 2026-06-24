"""
Unit tests for the plant disease prediction pipeline.
Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a 224x224 dummy leaf image for testing."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    img_path = tmp_path / "test_leaf.jpg"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def dummy_class_names():
    return [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Corn___Common_rust",
        "Potato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___healthy",
    ]


@pytest.fixture
def mock_model(dummy_class_names):
    """Mock Keras model that returns a valid softmax output."""
    model = MagicMock()
    num_classes = len(dummy_class_names)
    # Softmax-like output: all positive, sums to 1
    raw = np.random.rand(1, num_classes).astype(np.float32)
    raw = raw / raw.sum()
    model.predict.return_value = raw
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_output_shape(self, dummy_image_path):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {result.shape}"

    def test_output_dtype(self, dummy_image_path):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.dtype == np.float32

    def test_pixel_range(self, dummy_image_path):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.min() >= 0.0
        assert result.max() <= 255.0


class TestPredict:
    def test_returns_dict_with_required_keys(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "top3" in result
        assert "all_scores" in result

    def test_predicted_class_is_valid(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert result["predicted_class"] in dummy_class_names

    def test_confidence_between_0_and_1(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence {result['confidence']} is out of [0, 1] range"
        )

    def test_top3_has_three_items(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert len(result["top3"]) == 3

    def test_top3_confidences_descending(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        confs = [item["confidence"] for item in result["top3"]]
        assert confs == sorted(confs, reverse=True), "Top-3 should be sorted highest first"

    def test_all_scores_keys_match_class_names(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert set(result["all_scores"].keys()) == set(dummy_class_names)


class TestModelBuilding:
    def test_model_output_shape(self):
        from src.model import build_model
        model = build_model(num_classes=8)
        assert model.output_shape == (None, 8)

    def test_model_input_shape(self):
        from src.model import build_model
        model = build_model(num_classes=8)
        assert model.input_shape == (None, 224, 224, 3)

    def test_model_compiles(self):
        from src.model import build_model
        import tensorflow as tf
        model = build_model(num_classes=8)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        # If compile doesn't throw, we're good
        assert True
