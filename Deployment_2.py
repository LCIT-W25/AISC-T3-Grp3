import streamlit as st
from sentiment_analysis.run_model import predict_sentiment

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a review below and choose a model (RNN or GRU) to analyze its sentiment.")

# User input
user_input = st.text_area("Enter your review:", "")

# Model selection
model_choice = st.radio("Choose a model:", ("RNN", "GRU"))

# Prediction
if st.button("Analyze Sentiment"):
    if user_input.strip():
        model_type = "rnn" if model_choice == "RNN" else "gru"
        sentiment, score = predict_sentiment(user_input, model_type)
        
        st.subheader("Prediction Result:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {score:.4f}")
    else:
        st.warning("Please enter a review before analyzing.")

