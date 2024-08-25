from flask import Flask, request, jsonify
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

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
