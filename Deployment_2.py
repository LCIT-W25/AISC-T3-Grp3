import streamlit as st
from sentiment_analysis.run_model import predict_sentiment
from image_classification.run_image_model import predict_image_class

# Streamlit UI for both Sentiment Analysis and Image Classification
st.title("Sentiment & Image Classification App")

# Selection for the type of prediction (Sentiment or Image Classification)
prediction_type = st.radio(
    "Choose Prediction Type:",
    ("Sentiment Analysis",
     "Image Classification"))

if prediction_type == "Sentiment Analysis":
    # User input for sentiment analysis
    st.write(
        "Enter a review below and choose a model (RNN or GRU) to analyze its sentiment.")

    # Text input for review
    user_input = st.text_area("Enter your review:", "")

    # Model selection for Sentiment Analysis (RNN or GRU)
    model_choice = st.radio("Choose a model:", ("RNN", "GRU"))

    # Prediction
    if st.button("Predict Sentiment"):
        if user_input.strip():
            model_type = "rnn" if model_choice == "RNN" else "gru"
            sentiment, score = predict_sentiment(user_input, model_type)

            st.subheader("Prediction Result:")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** {score:.4f}")
        else:
            st.warning("Please enter a review before analyzing.")

elif prediction_type == "Image Classification":
    # User input for image classification
    st.write(
        "Upload an image and choose a model (CNN or Transfer Learning) to classify the image.")

    # Image upload
    image_file = st.file_uploader(
        "Choose an image...", type=[
            "jpg", "jpeg", "png"])

    # Model selection for Image Classification (CNN or Transfer Learning)
    model_choice = st.radio("Choose a model:", ("CNN", "Transfer Learning"))

    # Predict the image
    if st.button("Predict Image"):
        if image_file is not None:
            # Save the uploaded image to a temporary file
            with open("temp_image.jpg", "wb") as f:
                f.write(image_file.getbuffer())

            # Run prediction based on model choice
            if model_choice == "CNN":
                prediction = predict_image_class("temp_image.jpg", "cnn")
            else:
                prediction = predict_image_class(
                    "temp_image.jpg", "transfer_learning")

            st.subheader("Prediction Result:")
            st.write(f"Predicted Class: {prediction}")
        else:
            st.warning("Please upload an image to classify.")
