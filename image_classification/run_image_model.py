import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

# Paths for saved models
CNN_MODEL_PATH = "image_classification/cnn_model.h5"
TRANSFER_LEARNING_MODEL_PATH = "image_classification/transfer_learning_model.h5"

# Load models
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
transfer_learning_model = tf.keras.models.load_model(
    TRANSFER_LEARNING_MODEL_PATH)

# Function to preprocess image


def preprocess_image(img_path, target_size=(224, 224)):
    # Modify target_size based on your model
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if required
    return img_array

# Function to predict sentiment using a specified model


def predict_image_class(image_path, model_type="cnn"):
    img_array = preprocess_image(image_path)

    if model_type == "cnn":
        model = cnn_model
    elif model_type == "transfer_learning":
        model = transfer_learning_model
    else:
        raise ValueError(
            "Invalid model type. Choose 'cnn' or 'transfer_learning'.")

    prediction = model.predict(img_array)
    return prediction
