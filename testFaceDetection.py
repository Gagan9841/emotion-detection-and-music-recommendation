import cv2
import numpy as np
from keras.models import load_model

# Load the face detection and emotion classification models
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
    roi = np.expand_dims(roi, axis=0)     # Add batch dimension
    
    return roi

def detect_emotion(image_path):
    roi = preprocess_image(image_path)
    
    # Predict emotion
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    # cv2.imwrite("preprocessed.jpg", roi * 255)
    
    return label, float(emotion_probability)

if __name__ == "__main__":
    image_path = './test-image.jpg'
    emotion, confidence = detect_emotion(image_path)
    print(f"Detected emotion: {emotion}")
    print(f"Confidence: {confidence}")
