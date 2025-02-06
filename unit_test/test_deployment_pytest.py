import os
import sys
import pytest
import numpy as np

# import Deployment.py from parent folder
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

# Now import from Deployment.py
import Deployment
from Deployment import (
    preprocess_text,
    preprocess_image,
    dnn_model,
    knn_model,
    device,
    extract_resnet_features
)

# override Deployment.config with in-memory values
@pytest.fixture
def mock_config(monkeypatch):
    dummy_config = {
        "use_nb": True,
        "use_svm": True,
        "nb_version": "v1",
        "svm_version": "v2"
    }

    # Override the actual 'config' in Deployment.py
    monkeypatch.setattr(Deployment, "config", dummy_config)
    return dummy_config

# Test cases
@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("This is a Test!", "test"),
        ("Another?!! Example, text...", "another example text"),
        ("123 numbers 456", "numbers"),
    ],
)

def test_preprocess_text(input_text, expected):
    result = preprocess_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"


def test_preprocess_image():
    # Create a dummy 224x224 RGB image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_image)

    # shape should be (1, 224, 224, 1)
    assert processed.shape == (1, 224, 224, 1), f"Got shape {processed.shape}"
    # Check normalization
    assert 0 <= processed.min() and processed.max() <= 1, "Not normalized correctly."


def test_Deployment_with_mock_config(mock_config):

    # Ensures that our monkeypatched config overrides the file-based config.
    # The `Deployment.config` is replaced with dummy_config by the fixture
    assert Deployment.config["use_nb"] is True
    assert Deployment.config["use_svm"] is True
    assert Deployment.config["nb_version"] == "v1"
    assert Deployment.config["svm_version"] == "v2"
#
# @pytest.mark.skip(reason="Requires large model loads. Enable if you want integration tests.")
# def test_dnn_model_loading():
#     # Generate a dummy 224x224 RGB image and preprocess
#     dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
#     processed = preprocess_image(dummy_image)
#
#     # Run inference
#     predictions = dnn_model.predict(processed)
#
#     # Suppose your DNN outputs 5 classes
#     assert predictions.shape == (1, 5), (
#         f"Expected shape (1, 5) for predictions, got {predictions.shape}"
#     )


@pytest.mark.skip(reason="Requires large model loads. Enable if you want integration tests.")
def test_knn_model_inference():
    # Make a dummy 224x224 RGB image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Extract features using ResNet (Deployment.extract_resnet_features)
    features = extract_resnet_features(dummy_image)
    features = features.reshape(1, -1)  # shape (1, 512) typically for resnet18

    # Predict with kNN
    prediction = knn_model.predict(features)

    # Just check we got something valid (e.g., one label)
    assert len(prediction) == 1, f"Expected single prediction, got {prediction}"
