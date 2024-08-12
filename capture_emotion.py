import cv2
from deepface import DeepFace
from flask import Flask, request, jsonify

def detect_emotion(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'])
        return result['dominant_emotion']
    except ValueError as e:
        return f"Value Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

