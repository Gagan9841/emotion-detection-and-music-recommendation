from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import os
import numpy as np
from recommend_song import recommend_songs
from flask import Flask, request, jsonify
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

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
# model.load_weights('/var/www/html/8thproject/emotion-detection-and-music-recommendation/face_model.h5')

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
def get_recommendations():
    # Extract query parameters
    emotion = request.args.get('emotion')
    confidence = request.args.get('confidence', type=float)

    if not emotion or confidence is None:
        return jsonify({'error': 'Invalid input'}), 400

    # Get recommendations based on emotion
    try:
        recommendations = recommend_songs(emotion, num_recommendations=10, confidence=confidence)
        # Convert dataframe to list of dicts for JSON response
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify({"data": recommendations_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load the models
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = './Emotion_little_vgg.keras'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        raise ValueError("No face detected")
    
    # Select the largest face
    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = gray[fY:fY + fH, fX:fX + fW]
    
    # Resize to the required input shape of the emotion model
    roi = cv2.resize(roi, (48, 48))  # Resize to 48x48
    roi = roi.astype("float32") / 255.0  # Normalize
    roi = np.expand_dims(roi, axis=-1)    # Add channel dimension
    roi = np.expand_dims(roi, axis=0) 
    
    return roi

@app.route('/detect-emotion', methods=['POST'])
def save_image():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['files']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image
    save_path = './images/uploaded-image.jpg'
    file.save(save_path)
    
    try:
        roi = preprocess_image(save_path)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        return jsonify({'emotion': label, 'confidence': float(emotion_probability)})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.makedirs('images')
    app.run(debug=True)