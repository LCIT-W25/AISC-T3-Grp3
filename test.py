import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# =============== UTILITY FUNCTIONS FOR LOADING ===============

@st.cache_resource
def load_model(path):
    """Load a Keras model from the given path."""
    return tf.keras.models.load_model(path)


@st.cache_data
def load_pickle(path):
    """Load a pickle file from the given path."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


# =============== LOAD MODELS AND LABEL ENCODERS ===============

# --- Sentiment Analysis ---
rnn_model = load_model("sentiment_analysis/rnn_model.h5")
gru_model = load_model("sentiment_analysis/gru_model.h5")

rnn_tokenizer = load_pickle("sentiment_analysis/rnn_tokenizer.pickle")
gru_tokenizer = load_pickle("sentiment_analysis/gru_tokenizer.pkl")

# IMPORTANT: Load label encoders for sentiment (multi-class scenario)
rnn_label_encoder = load_pickle("sentiment_analysis/rnn_label_encoder.pickle")
gru_label_encoder = load_pickle("sentiment_analysis/rnn_label_encoder.pickle")

# --- Image Classification ---
cnn_model = load_model("image_classification/cnn_model.h5")
transfer_model = load_model("image_classification/efficientnet_transfer_model.h5")

cnn_label_encoder = load_pickle("image_classification/label_encoder_cnn.pkl")
efficientnet_label_encoder = load_pickle("image_classification/label_encoder_efficientnet.pkl")


# =============== PREDICTION FUNCTIONS ===============

def predict_sentiment(text, model, tokenizer, label_encoder, max_len=100):
    """
    Predict the class label and confidence for a given text
    using the specified model, tokenizer, and label encoder.

    Assumes the model outputs a probability distribution
    over multiple classes (shape: [batch_size, num_classes]).
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Model prediction -> shape: (1, num_classes)
    predictions = model.predict(padded)

    # Optional: Display raw probabilities
    # st.write("Raw sentiment model output:", predictions[0])

    # Get predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # Convert index to label
    label = label_encoder.inverse_transform([class_idx])[0]
    return label, float(confidence)


def predict_image(img, model, label_encoder):
    """
    Predict the class label and confidence for a given PIL image
    using the specified Keras model and label encoder.
    """
    # Resize and scale the image
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction -> shape: (1, num_classes)
    predictions = model.predict(img_array)

    # Optional: Display raw probabilities
    # st.write("Raw image model output:", predictions[0])

    # Identify predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # Convert the index to a label using the label encoder
    label = label_encoder.inverse_transform([class_idx])[0]
    return label, float(confidence)


# =============== STREAMLIT UI ===============

st.title("AI Model Deployment")

# ---------- Sentiment Analysis ----------
st.header("Sentiment Analysis")
text_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    # RNN Prediction
    rnn_label, rnn_conf = predict_sentiment(text_input, rnn_model, rnn_tokenizer, rnn_label_encoder)
    st.write(f"**RNN Model Prediction:** {rnn_label} (Confidence: {rnn_conf:.2f})")

    # GRU Prediction
    gru_label, gru_conf = predict_sentiment(text_input, gru_model, gru_tokenizer, gru_label_encoder)
    st.write(f"**GRU Model Prediction:** {gru_label} (Confidence: {gru_conf:.2f})")

# ---------- Image Classification ----------
st.header("Image Classification")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    img = load_img(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image"):
        # CNN Model
        cnn_label, cnn_conf = predict_image(img, cnn_model, cnn_label_encoder)
        st.write(f"**CNN Model Prediction:** {cnn_label} (Confidence: {cnn_conf:.2f})")

        # Transfer (EfficientNet) Model
        eff_label, eff_conf = predict_image(img, transfer_model, efficientnet_label_encoder)
        st.write(f"**EfficientNet Prediction:** {eff_label} (Confidence: {eff_conf:.2f})")
