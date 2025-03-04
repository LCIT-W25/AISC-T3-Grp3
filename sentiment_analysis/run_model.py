import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths for saved models and tokenizers
RNN_MODEL_PATH = "sentiment_analysis/rnn_model.h5"
GRU_MODEL_PATH = "sentiment_analysis/gru_model.h5"
RNN_TOKENIZER_PATH = "sentiment_analysis/rnn_tokenizer.pkl"
GRU_TOKENIZER_PATH = "sentiment_analysis/gru_tokenizer.pkl"

# Load tokenizers
with open(RNN_TOKENIZER_PATH, "rb") as file:
    rnn_tokenizer = pickle.load(file)

with open(GRU_TOKENIZER_PATH, "rb") as file:
    gru_tokenizer = pickle.load(file)

# Load models
rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)

# Function to preprocess text
def preprocess_text(text, model_type="rnn", max_len=100):
    if model_type == "rnn":
        tokenizer = rnn_tokenizer
    elif model_type == "gru":
        tokenizer = gru_tokenizer
    else:
        raise ValueError("Invalid model type. Choose 'rnn' or 'gru'.")

    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding="post")
    return padded

# Function to predict sentiment using a specified model
def predict_sentiment(text, model_type="rnn"):
    processed_text = preprocess_text(text, model_type)

    if model_type == "rnn":
        model = rnn_model
    elif model_type == "gru":
        model = gru_model
    else:
        raise ValueError("Invalid model type. Choose 'rnn' or 'gru'.")

    prediction = model.predict(processed_text)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Example usage
if __name__ == "__main__":
    text = "The product quality is terrible and I hate it!"
    
    sentiment_rnn, score_rnn = predict_sentiment(text, "rnn")
    sentiment_gru, score_gru = predict_sentiment(text, "gru")
    
    print(f"RNN Sentiment: {sentiment_rnn}, Score: {score_rnn:.4f}")
    print(f"GRU Sentiment: {sentiment_gru}, Score: {score_gru:.4f}")
