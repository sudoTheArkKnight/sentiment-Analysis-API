from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch
import os

# Initialize the Flask app
app = Flask(__name__)

# Load a sentiment-analysis pipeline using a pre-trained BERT model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=device)


# Route for the home page to serve the front end
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment():
    # Get the input JSON data
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input, please provide text field'}), 400

    text = data['text']

    try:
        # Perform sentiment analysis on the input text
        analysis = sentiment_analyzer(text)[0]

        # Convert model's output to positive/negative with percentage
        label = analysis['label']
        score = analysis['score'] * 100

        # Classify into Positive or Negative
        if label.lower() in ['1 star', '2 stars', 'negative']:
            sentiment = 'negative'
        else:
            sentiment = 'positive'

        response = {
            'sentiment': sentiment,
            'confidence': round(score, 2)
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define a health check endpoint to confirm service status
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'up'}), 200


# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
