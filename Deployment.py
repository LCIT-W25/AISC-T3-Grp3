import streamlit as st
import requests
import joblib
import tensorflow as tf
import numpy as np
import cv2
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_tokens)


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet for feature extraction
resnet = models.resnet18(pretrained=True)
resnet.eval().to(device)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)

# Image transformation pipeline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

use_nb = config["use_nb"]
use_svm = config["use_svm"]
nb_version = config["nb_version"]
svm_version = config["svm_version"]

# MODEL VERSION SWITCH
st.write("### Model Version Switcher")
selected_nb_version = st.radio(
    "Naïve Bayes Version",
    ["v1", "v2"],
    index=["v1", "v2"].index(nb_version)
)
selected_svm_version = st.radio(
    "SVM Version",
    ["v1", "v2"],
    index=["v1", "v2"].index(svm_version)
)

if (selected_nb_version != nb_version) or (selected_svm_version != svm_version):
    config["nb_version"] = selected_nb_version
    config["svm_version"] = selected_svm_version

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    st.warning("Configuration updated. Please manually re-run the app to load the new versions.")

# Load models according to the current versions
if use_nb:
    nb_model = joblib.load(f"naive_bayes_{nb_version}.pkl")
if use_svm:
    svm_model = joblib.load(f"svm_{svm_version}.pkl")

vectorizer = joblib.load("vectorizer.pkl")
dnn_model = tf.keras.models.load_model("DNN.keras")
knn_model = joblib.load("knn.joblib")

# Streamlit UI
st.title("AI-Powered Text & Image Classification")

# Text Classification Section
st.header("Review Sentiment Analysis")

review_text = st.text_area("Enter a review:", "")
if st.button("Analyze Sentiment"):
    if review_text:
        cleaned_text = preprocess_text(review_text)
        transformed_text = vectorizer.transform([cleaned_text])

        if use_nb:
            nb_prediction = nb_model.predict(transformed_text)[0]
            st.write(f"**Naïve Bayes {nb_version} Prediction:** {'Positive' if nb_prediction else 'Negative'}")

        if use_svm:
            svm_prediction = svm_model.predict(transformed_text)[0]
            st.write(f"**SVM {svm_version} Prediction:** {'Positive' if svm_prediction else 'Negative'}")
    else:
        st.warning("Please enter a review.")

# Image Classification Section
st.header("Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
    image = np.expand_dims(image, axis=-1)  # channel dimension
    image = np.array(image) / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # batch dimension
    return image


def extract_resnet_features(image):
    """Extracts a 512-dim feature vector using ResNet18."""
    image = Image.fromarray(image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(tensor)  # [1, 512, 1, 1]
    return features.view(-1).cpu().numpy()


class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image"):
        processed_dnn = preprocess_image(image)
        dnn_prediction = np.argmax(dnn_model.predict(processed_dnn))
        dnn_label = class_labels[dnn_prediction]

        knn_features = extract_resnet_features(image)
        knn_features = knn_features.reshape(1, -1)
        knn_prediction = knn_model.predict(knn_features)[0]

        st.write(f"**DNN Model Prediction:** {dnn_label}")
        st.write(f"**kNN Model Prediction:** {knn_prediction}")
