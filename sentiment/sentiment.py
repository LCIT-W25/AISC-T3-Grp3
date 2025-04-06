import os
import torch
import joblib
import re
import emoji
import string
import numpy as np
import pandas as pd
import shap
from transformers import XLNetForSequenceClassification, XLNetTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths (adjust paths as needed)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "my_best_xlnet_model")
tokenizer_path = os.path.join(base_dir, "my_best_xlnet_tokenizer")
label_encoder_path = os.path.join(base_dir, "label_encoder.joblib")

# Load model, tokenizer, and label encoder
model = XLNetForSequenceClassification.from_pretrained(model_path)
tokenizer = XLNetTokenizer.from_pretrained(tokenizer_path)
le = joblib.load(label_encoder_path)

model.to(device)
model.eval()


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    text = text.lower()
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_sentiment(text: str) -> str:
    """Predict sentiment for a given text without SHAP explanation."""
    processed_text = preprocess_text(text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=-1).item()
    sentiment_label = le.inverse_transform([pred_class])[0]
    return sentiment_label


def analyze_text_with_shap(text: str, top_n: int = 3):
    """
    Predict sentiment and compute SHAP explanations for the input text.

    Returns:
      predicted_sentiment: the predicted sentiment label
      top_tokens_df: a DataFrame with the top N tokens and their SHAP values
    """
    processed_text = preprocess_text(text)

    # Get prediction and predicted class id
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_class_id = torch.argmax(outputs.logits, dim=-1).item()
    predicted_sentiment = le.inverse_transform([pred_class_id])[0]

    # Define a prediction function for SHAP that works on a list of texts.
    def shap_predict(texts):
        texts = [str(t) for t in texts]  # Ensure every element is a string
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits.cpu().detach().numpy()

    # Create a text masker for SHAP using the tokenizer
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(shap_predict, masker)

    # Compute SHAP values for the input text
    shap_values = explainer([text])

    # Extract SHAP values and tokens for the predicted class
    token_shap_values = shap_values.values[0][:, pred_class_id]  # shape: (num_tokens,)
    tokens = shap_values.data[0]

    # Get top N tokens based on absolute SHAP contribution
    abs_values = np.abs(token_shap_values)
    sorted_indices = np.argsort(abs_values)[::-1][:top_n]
    top_tokens = [tokens[i] for i in sorted_indices]
    top_contributions = [token_shap_values[i] for i in sorted_indices]

    top_tokens_df = pd.DataFrame({
        "Token": top_tokens,
        "SHAP Value": top_contributions
    })

    return predicted_sentiment, top_tokens_df
