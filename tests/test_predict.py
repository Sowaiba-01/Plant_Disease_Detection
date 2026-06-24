
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a 224x224 dummy RGB image for testing — no real leaf needed."""
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
    """Mock Keras model that returns a valid softmax-like output (sums to 1)."""
    model = MagicMock()
    num_classes = len(dummy_class_names)
    raw = np.random.rand(1, num_classes).astype(np.float32)
    raw = raw / raw.sum()                  # normalise to sum to 1
    model.predict.return_value = raw
    return model


@pytest.fixture
def high_confidence_mock_model(dummy_class_names):
    """Mock model that always returns very high confidence on class 0."""
    model = MagicMock()
    num_classes = len(dummy_class_names)
    scores = np.zeros((1, num_classes), dtype=np.float32)
    scores[0, 0] = 0.99
    scores[0, 1:] = 0.01 / (num_classes - 1)
    model.predict.return_value = scores
    return model


@pytest.fixture
def low_confidence_mock_model(dummy_class_names):
    """Mock model that returns nearly uniform (low confidence) output."""
    model = MagicMock()
    num_classes = len(dummy_class_names)
    scores = np.full((1, num_classes), 1.0 / num_classes, dtype=np.float32)
    model.predict.return_value = scores
    return model


# ── preprocess_image ───────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_output_shape(self, dummy_image_path):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.shape == (1, 224, 224, 3), (
            f"Expected (1, 224, 224, 3), got {result.shape}"
        )

    def test_output_dtype_is_float32(self, dummy_image_path):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.dtype == np.float32

    def test_pixel_values_in_0_255_range(self, dummy_image_path):
        """EfficientNetB0 expects [0, 255] — NOT divided by 255."""
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_path)
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_handles_non_rgb_image(self, tmp_path):
        """RGBA images should be converted to RGB without crashing."""
        from src.predict import preprocess_image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA"
        )
        img_path = str(tmp_path / "rgba.png")
        img.save(img_path)
        result = preprocess_image(img_path)
        assert result.shape == (1, 224, 224, 3)


# ── predict ────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_dict_with_required_keys(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        for key in ("predicted_class", "confidence", "top3", "all_scores", "low_confidence"):
            assert key in result, f"Missing key: {key}"

    def test_predicted_class_is_valid_or_unknown(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert result["predicted_class"] in dummy_class_names or result["predicted_class"] == "Unknown"

    def test_confidence_between_0_and_1(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence {result['confidence']} out of [0, 1]"
        )

    def test_top3_has_exactly_three_items(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert len(result["top3"]) == 3

    def test_top3_sorted_descending(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        confs = [item["confidence"] for item in result["top3"]]
        assert confs == sorted(confs, reverse=True), "Top-3 must be sorted highest → lowest"

    def test_all_scores_has_all_classes(self, dummy_image_path, mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=mock_model, class_names=dummy_class_names)
        assert set(result["all_scores"].keys()) == set(dummy_class_names)

    def test_high_confidence_not_flagged(self, dummy_image_path, high_confidence_mock_model, dummy_class_names):
        from src.predict import predict
        result = predict(dummy_image_path, model=high_confidence_mock_model, class_names=dummy_class_names)
        assert result["low_confidence"] is False
        assert result["predicted_class"] != "Unknown"

    def test_low_confidence_flagged_as_unknown(self, dummy_image_path, low_confidence_mock_model, dummy_class_names):
        """When confidence is below threshold, predicted_class must be 'Unknown'."""
        from src.predict import predict
        result = predict(dummy_image_path, model=low_confidence_mock_model, class_names=dummy_class_names)
        assert result["low_confidence"] is True
        assert result["predicted_class"] == "Unknown"

    def test_missing_image_raises_error(self, mock_model, dummy_class_names):
        from src.predict import predict
        with pytest.raises(Exception):
            predict("/nonexistent/path/leaf.jpg", model=mock_model, class_names=dummy_class_names)


# ── model building (weights mocked to avoid 29MB download in CI) ──────────────

class TestModelBuilding:
    def test_model_output_shape(self):
        """FIX: use load_weights=False to skip ImageNet download in CI/tests."""
        from src.model import build_model
        model = build_model(num_classes=8, load_weights=False)
        assert model.output_shape == (None, 8)

    def test_model_input_shape(self):
        from src.model import build_model
        model = build_model(num_classes=8, load_weights=False)
        assert model.input_shape == (None, 224, 224, 3)

    def test_model_compiles_without_error(self):
        import tensorflow as tf
        from src.model import build_model
        model = build_model(num_classes=8, load_weights=False)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def test_custom_num_classes(self):
        from src.model import build_model
        for n in [2, 10, 38]:
            model = build_model(num_classes=n, load_weights=False)
            assert model.output_shape == (None, n), f"Failed for num_classes={n}"

    def test_custom_input_shape(self):
        from src.model import build_model
        model = build_model(num_classes=8, input_shape=(128, 128, 3), load_weights=False)
        assert model.input_shape == (None, 128, 128, 3)


# ── load_model_and_classes ─────────────────────────────────────────────────────

class TestLoadModelAndClasses:
    def test_missing_model_raises_file_not_found(self, tmp_path):
        from src.predict import load_model_and_classes
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model_and_classes(
                model_path=tmp_path / "nonexistent.keras",
                class_names_path=tmp_path / "classes.json",
            )

    def test_missing_class_names_raises_file_not_found(self, tmp_path):
        from src.predict import load_model_and_classes
        # Create a dummy model file so the first check passes
        fake_model = tmp_path / "model.keras"
        fake_model.write_text("fake")
        with pytest.raises(FileNotFoundError, match="Class names not found"):
            load_model_and_classes(
                model_path=fake_model,
                class_names_path=tmp_path / "nonexistent.json",
            )
