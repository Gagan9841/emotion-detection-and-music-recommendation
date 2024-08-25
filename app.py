from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from capture_emotion import detect_emotion
from PIL import Image
import os
import numpy as np
from recommend_song import recommend_songs, load_data
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

# Load the models
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400
    
    image_file = request.files['image']
    if not image_file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Convert image file to array
    image = Image.open(image_file)
    image = image.convert('RGB')
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

    return jsonify({'emotion': label, 'confidence': float(emotion_probability)})

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
