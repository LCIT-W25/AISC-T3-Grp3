import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import shap

# Load models and tokenizers
#@st.cache_resource
def load_rnn_model():
    return tf.keras.models.load_model("sentiment_analysis/rnn_model.h5")

#@st.cache_resource
def load_gru_model():
    return tf.keras.models.load_model("sentiment_analysis/gru_model.h5")

#@st.cache_data
def load_tokenizer(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

#@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("image_classification/cnn_model.h5")

#@st.cache_resource
def load_transfer_model():
    return tf.keras.models.load_model("image_classification/efficientnet_transfer_model.h5")

def load_label_encoder(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

rnn_model = load_rnn_model()
gru_model = load_gru_model()
rnn_tokenizer = load_tokenizer("sentiment_analysis/rnn_tokenizer.pickle")
gru_tokenizer = load_tokenizer("sentiment_analysis/gru_tokenizer.pkl")
cnn_model = load_cnn_model()
transfer_model = load_transfer_model()
label_encoder = load_label_encoder("image_classification/label_encoder_efficientnet.pkl")

# Sentiment Analysis Function
def predict_sentiment(text, model, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative", prediction

# Image Classification Function
def predict_image(img, model):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_idx, confidence

# Streamlit UI
st.title("AI Model Deployment")

# Sentiment Analysis
st.header("Sentiment Analysis")
text_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    rnn_result, rnn_conf = predict_sentiment(text_input, rnn_model, rnn_tokenizer)
    gru_result, gru_conf = predict_sentiment(text_input, gru_model, gru_tokenizer)
    st.write(f"**RNN Prediction:** {rnn_result} (Confidence: {rnn_conf:.2f})")
    st.write(f"**GRU Prediction:** {gru_result} (Confidence: {gru_conf:.2f})")

# Image Classification
st.header("Image Classification")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    img = load_img(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Image"):
        cnn_class, cnn_conf = predict_image(img, cnn_model)
        transfer_class, transfer_conf = predict_image(img, transfer_model)
        st.write(f"**CNN Model Prediction:** Class {cnn_class} (Confidence: {cnn_conf:.2f})")
        
        # Convert class index to class label using the label encoder
        transfer_class_label = label_encoder.inverse_transform([transfer_class])[0]
        st.write(f"**Transfer Learning Prediction:** {transfer_class_label} (Confidence: {transfer_conf:.2f})")
