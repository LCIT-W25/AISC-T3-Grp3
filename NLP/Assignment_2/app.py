import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained Naive Bayes model
nb_model = joblib.load('naive_bayes_model.pkl')

# Load the trained TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Save the vectorizer during training


@app.route('/predict-review', methods=['POST'])
def predict_review():
    data = request.json['text']  # Getting text input (expecting a list of strings)

    # Vectorize the input text using the loaded TF-IDF vectorizer
    vectorized_data = tfidf_vectorizer.transform(data)  # Transform into 2D array

    # Make predictions using the Naive Bayes model
    prediction = nb_model.predict(vectorized_data)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
