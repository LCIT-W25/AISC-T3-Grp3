import pytest
import os
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from PIL import Image
import tempfile

from Deploy import (
    download_cnn_model,
    cleanup_model,
    predict_sentiment,
    predict_image
)


@patch("builtins.open", new_callable=mock_open,
       read_data="https://drive.google.com/file/d/16Vwm2aar9Qo0e1ZvJkUM0GPpG402cekI/view?usp=sharing")
@patch("os.path.exists", return_value=False)
@patch("re.search")
@patch("gdown.download", return_value="image_classification/cnn_model.h5")
def test_download_cnn_model(
        mock_gdown,
        mock_re_search,
        mock_exists,
        mock_file):
    # Mock the regex match so it returns a group that looks like a file ID
    mock_re_search.return_value.group.return_value = "16Vwm2aar9Qo0e1ZvJkUM0GPpG402cekI"

    path = download_cnn_model()
    assert path == "image_classification/cnn_model.h5"
    mock_gdown.assert_called_once()
    mock_re_search.assert_called_once()


def test_cleanup_model():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_file_path = tmp.name
    # Confirm the file exists before cleanup
    assert os.path.exists(temp_file_path) is True
    cleanup_model(temp_file_path)
    # Now it should be gone
    assert not os.path.exists(temp_file_path)


def test_predict_image():

    # Create a mock model and assign its predict method manually.
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=np.array([[0.8, 0.2]]))

    # Create a mock label encoder that returns a label for index 0.
    mock_label_encoder = MagicMock()
    mock_label_encoder.inverse_transform.return_value = ["class_0"]

    # Create a dummy image in memory.
    img = Image.new("RGB", (300, 300), color="red")

    # Call the function.
    label, conf = predict_image(img, mock_model, mock_label_encoder)

    # With the patched predict output of [[0.8, 0.2]],
    # np.argmax returns 0 and confidence should be 0.8.
    assert label == "class_0"
    assert conf == 0.8
    mock_label_encoder.inverse_transform.assert_called_with([0])


def test_predict_sentiment():

    # Create a mock model and assign its predict method manually.
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=np.array([[0.1, 0.9]]))

    # Create a mock tokenizer that returns a dummy sequence.
    mock_tokenizer = MagicMock()
    mock_tokenizer.texts_to_sequences.return_value = [[5, 10, 15]]

    # Create a mock label encoder that returns a label for index 1.
    mock_label_encoder = MagicMock()
    mock_label_encoder.inverse_transform.return_value = ["class_1"]

    text = "hi is this good food"

    # Call the function.
    label, conf = predict_sentiment(
        text,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
        max_len=10
    )

    # With the patched predict output of [[0.1, 0.9]],
    # np.argmax returns 1 and confidence should be 0.9.
    assert label == "class_1"
    assert conf == 0.9

    # Ensure the tokenizer was called correctly.
    mock_tokenizer.texts_to_sequences.assert_called_with([text])
