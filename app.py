from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from capture_emotion import detect_emotion
from PIL import Image
import os
import numpy as np
from recommend_song import recommend_songs, load_data

app = Flask(__name__)

CORS(app)  # This will enable CORS for all routes

import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model


# Create the model and load weights
model = create_model()
# model.load_weights('/var/www/html/8thproject/emotion-detection-and-music-recommendation/trained_model_weights.weights.h5')
model.load_weights('/var/www/html/8thproject/emotion-detection-and-music-recommendation/face_model.h5')

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['files']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        image = Image.open(file.stream).convert('L')  # Ensure image is in grayscale
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions
        
        # Make prediction
        prediction = model.predict(image_array)
        
        return jsonify({'emotion': prediction[0].tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/recommend-songs', methods=['GET'])
def recommend_songs_endpoint():
    try:
        # Extract emotion and num_recommendations from query parameters
        emotion = request.args.get('emotion')
        num_recommendations = int(request.args.get('num_recommendations', 10))  # Default to 10 if not provided
        
        # Load the dataset
        df = load_data()
        
        # Get recommendations
        recommendations = recommend_songs(emotion, df, num_recommendations)
        
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
