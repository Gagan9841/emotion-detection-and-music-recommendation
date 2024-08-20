from flask import Flask, request, jsonify
import joblib
from capture_emotion import detect_emotion
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model_weights.weights.h5')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'emotion': prediction[0]})

@app.route('/capture-emotion', methods=['POST'])
def capture_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = os.path.join("static", image.filename)

    # Ensure the directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    image.save(image_path)
    emotion = detect_emotion(image_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
