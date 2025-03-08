import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import atexit
import os
import re
import gdown
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# READ THE GOOGLE DRIVE LINK FROM TEXT FILE & DOWNLOAD CNN MODEL


def download_cnn_model():

    text_file_path = "image_classification/link for CNN Model.txt"
    with open(text_file_path, "r") as f:
        link = f.read().strip()

    # Attempt to extract the file ID from a link like:
    # https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    match = re.search(r'/d/(.*?)/', link)
    if not match:
        st.error("Could not extract file ID from the provided link.")
        return None

    file_id = match.group(1)
    direct_link = f"https://drive.google.com/uc?id={file_id}"

    # Define where we want to save the downloaded model
    output_path = "image_classification/cnn_model.h5"

    # Download only if it isn't already downloaded
    if not os.path.exists(output_path):
        st.write("Downloading CNN model from Google Drive...")
        gdown.download(direct_link, output_path, quiet=False)

    return output_path

# CLEANUP FUNCTION TO REMOVE DOWNLOADED MODEL ON EXIT


def cleanup_model(path):
    """Remove the model file if it exists."""
    if path and os.path.exists(path):
        os.remove(path)


# DOWNLOAD THE CNN MODEL & REGISTER CLEANUP
cnn_model_path = download_cnn_model()  # Download the CNN model at startup
atexit.register(lambda: cleanup_model(cnn_model_path))

# LOAD ALL MODELS, TOKENIZERS, AND LABEL ENCODERS


@st.cache_resource
def load_cnn_model(path):
    """Load the CNN model from the given path."""
    return tf.keras.models.load_model(path)


if cnn_model_path:
    cnn_model = load_cnn_model(cnn_model_path)
else:
    cnn_model = None


@st.cache_resource
def load_model(path):
    """Generic function to load a Keras model from a local path."""
    return tf.keras.models.load_model(path)


# --- Other local models (EfficientNet, RNN, GRU) ---
transfer_model = load_model(
    "image_classification/efficientnet_transfer_model.h5")
rnn_model = load_model("sentiment_analysis/rnn_model.h5")
gru_model = load_model("sentiment_analysis/gru_model.h5")


# --- Function to load pickle files (tokenizers, label encoders, etc.) ---
@st.cache_data
def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


# --- Sentiment tokenizers ---
rnn_tokenizer = load_pickle("sentiment_analysis/rnn_tokenizer.pickle")
gru_tokenizer = load_pickle("sentiment_analysis/gru_tokenizer.pkl")

# --- Sentiment label encoders ---
# (Adjust paths if you have separate pickles for RNN vs GRU)
rnn_label_encoder = load_pickle("sentiment_analysis/rnn_label_encoder.pickle")
gru_label_encoder = load_pickle("sentiment_analysis/rnn_label_encoder.pickle")

# --- Image label encoders ---
cnn_label_encoder = load_pickle("image_classification/label_encoder_cnn.pkl")
efficientnet_label_encoder = load_pickle(
    "image_classification/label_encoder_efficientnet.pkl")

# --- NEW: Image tokenizers (if truly needed by your pipeline) ---
cnn_tokenizer = load_pickle("image_classification/tokenizer_cnn.pkl")
efficientnet_tokenizer = load_pickle(
    "image_classification/tokenizer_efficientnet.pkl")


# DEFINE PREDICTION FUNCTIONS
def predict_sentiment(text, model, tokenizer, label_encoder, max_len=100):

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Model prediction -> shape: (1, num_classes)
    predictions = model.predict(padded)

    # Identify predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # Convert index to label
    label = label_encoder.inverse_transform([class_idx])[0]
    return label, float(confidence)


def predict_image(img, model, label_encoder):

    # Standard image preprocessing
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction -> shape: (1, num_classes)
    predictions = model.predict(img_array)

    # Identify predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # Convert the index to a label using the label encoder
    label = label_encoder.inverse_transform([class_idx])[0]
    return label, float(confidence)


# STREAMLIT UI

st.title("AI Model Deployment with Tokenizers & Temporary CNN Model Download")

# ---------- Sentiment Analysis ----------
st.header("Sentiment Analysis")
text_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    # RNN Model Prediction
    if rnn_model is not None:
        rnn_label, rnn_conf = predict_sentiment(
            text_input,
            rnn_model,
            rnn_tokenizer,
            rnn_label_encoder
        )
        st.write(
            f"**RNN Model Prediction:** {rnn_label} (Confidence: {rnn_conf:.2f})")
    else:
        st.warning("RNN model not loaded.")

    # GRU Model Prediction
    if gru_model is not None:
        gru_label, gru_conf = predict_sentiment(
            text_input,
            gru_model,
            gru_tokenizer,
            gru_label_encoder
        )
        st.write(
            f"**GRU Model Prediction:** {gru_label} (Confidence: {gru_conf:.2f})")
    else:
        st.warning("GRU model not loaded.")

# ---------- Image Classification ----------
st.header("Image Classification")
uploaded_image = st.file_uploader(
    "Upload an image", type=[
        "jpg", "png", "jpeg"])

if uploaded_image:
    img = load_img(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image"):
        # CNN Model
        if cnn_model is not None:
            cnn_label, cnn_conf = predict_image(
                img,
                cnn_model,
                cnn_label_encoder,
                tokenizer=cnn_tokenizer  # <-- Use CNN tokenizer
            )
            st.write(
                f"**CNN Model Prediction:** {cnn_label} (Confidence: {cnn_conf:.2f})")
        else:
            st.warning(
                "CNN model not loaded. Check download link or file paths.")

        # EfficientNet Model
        if transfer_model is not None:
            eff_label, eff_conf = predict_image(
                img,
                transfer_model,
                efficientnet_label_encoder,
                tokenizer=efficientnet_tokenizer  # <-- Use EfficientNet tokenizer
            )
            st.write(
                f"**EfficientNet Prediction:** {eff_label} (Confidence: {eff_conf:.2f})")
        else:
            st.warning("EfficientNet model not loaded.")
