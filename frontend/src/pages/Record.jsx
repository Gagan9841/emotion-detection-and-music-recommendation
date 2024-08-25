import React, { useState, useRef } from "react";
import axios from "axios";
import Navbar from "../components/Navbar";
import { useSelector } from "react-redux";

const Record = (props) => {
  const [predictedEmotion, setPredictedEmotion] = useState(null);
  const [recommendedSongs, setRecommendedSongs] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const token = useSelector((state) => state.user.token);
  const [capturedImage, setCapturedImage] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startCamera = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        setCameraOn(true);
      })
      .catch((err) => {
        console.error("Error accessing camera: ", err);
      });
  };

  const captureImage = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = 48; // Set to model's expected width
    canvas.height = 48; // Set to model's expected height
    const context = canvas.getContext("2d");

    // Draw the video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the image to grayscale
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const avg =
        (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
      imageData.data[i] = avg; // Red channel
      imageData.data[i + 1] = avg; // Green channel
      imageData.data[i + 2] = avg; // Blue channel
    }
    context.putImageData(imageData, 0, 0);

    // Convert the canvas content to a blob
    canvas.toBlob((blob) => {
      setCapturedImage(blob);
      playSound();
      stopCamera();
    }, "image/png");
  };

  const playSound = () => {
    const audioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // A4 note
    oscillator.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.5); // Play sound for 0.5 seconds
  };

  const stopCamera = () => {
    const stream = videoRef.current.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
    setCameraOn(false);
  };
  const fetchRecommendedSongs = async (emotion) => {
    try {
      const params = new URLSearchParams();
      emotion.forEach((value, index) => {
        params.append(`emotion`, value);
      });

      const response = await axios.get(
        `http://127.0.0.1:5000/recommend-songs?${params.toString()}`
      );

      setRecommendedSongs(response.data.songs);
    } catch (error) {
      console.error("Error fetching recommended songs:", error);
    }
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("files", capturedImage, "image.png");

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict-emotion",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data.error) {
        alert(`Error: ${response.data.error}`);
      } else {
        // Determine the emotion with the highest probability
        const emotionProbabilities = response.data.emotion;
        const emotionLabels = ["sad", "happy", "energetic", "calm"];
        const highestProbabilityIndex = emotionProbabilities.indexOf(
          Math.max(...emotionProbabilities)
        );
        const detectedEmotion = emotionLabels[highestProbabilityIndex];

        setPredictedEmotion(detectedEmotion);
        fetchRecommendedSongs(detectedEmotion);
      }
    } catch (error) {
      console.error("Error making prediction:", error);
    }
  };

  return (
    <>
      <Navbar />
      <div className="w-full h-full flex flex-col items-center justify-center mt-8">
        {/* Camera Feed */}
        <video ref={videoRef} autoPlay className="w-full max-w-md" />
        <canvas ref={canvasRef} className="w-full max-w-md mt-4" />

        {/* Start Camera Button */}
        {!cameraOn && (
          <div
            className="bg-blue-500 hover:bg-blue-600 px-6 py-3 mt-4 rounded-full text-white cursor-pointer transition-all duration-300"
            onClick={startCamera}
          >
            Start Camera
          </div>
        )}

        {/* Capture Image Button */}
        <div
          className="bg-green-500 hover:bg-green-600 px-6 py-3 mt-4 rounded-full text-white cursor-pointer transition-all duration-300"
          onClick={captureImage}
        >
          Capture Image
        </div>

        {/* Predict Button (Visible only after capturing) */}
        {capturedImage && (
          <div
            className="bg-rose-500 hover:bg-rose-600 px-6 py-3 mt-4 rounded-full text-white cursor-pointer transition-all duration-300"
            onClick={handlePredict}
          >
            Predict
          </div>
        )}

        {/* Display Predicted Emotion */}
        {predictedEmotion && (
          <div className="mt-8 text-2xl text-rose-500 font-normal">
            Decode the Feels: Emotion Unveiled - {predictedEmotion}
          </div>
        )}

        {/* Display Recommended Songs */}
        {recommendedSongs.length > 0 && (
          <div className="mt-8 text-xl text-blue-500 font-normal">
            <h2>Recommended Songs:</h2>
            <ul>
              {recommendedSongs.map((song, index) => (
                <li key={index}>{song}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </>
  );
};

export default Record;
