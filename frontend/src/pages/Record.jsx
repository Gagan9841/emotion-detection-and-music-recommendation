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

  const stopCamera = () => {
    const stream = videoRef.current.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
    setCameraOn(false);
  };

  const captureImage = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

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

  const fetchRecommendedSongs = async (emotion, confidence) => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:5000/recommend-songs`,
        {
          params: {
            emotion: emotion,
            confidence: confidence,
          },
        }
      );

      setRecommendedSongs(response.data.data); // Adjust based on actual response structure
    } catch (error) {
      console.error("Error fetching recommended songs:", error);
    }
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("files", capturedImage);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/detect-emotion",
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
        const detectedEmotion = response.data.emotion;
        const confidence = response.data.confidence;

        setPredictedEmotion(detectedEmotion);
        await fetchRecommendedSongs(detectedEmotion, confidence);
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
        <video
          ref={videoRef}
          autoPlay
          className="w-full max-w-md border-2 border-gray-300 rounded-lg"
        />
        <canvas
          ref={canvasRef}
          className="w-full max-w-md mt-4 border-2 border-gray-300 rounded-lg"
        />

        {/* Start/Stop Camera Button */}
        {!cameraOn ? (
          <button
            className="bg-blue-500 hover:bg-blue-600 px-6 py-3 mt-4 rounded-full text-white transition-all duration-300"
            onClick={startCamera}
          >
            Start Camera
          </button>
        ) : (
          <button
            className="bg-red-500 hover:bg-red-600 px-6 py-3 mt-4 rounded-full text-white transition-all duration-300"
            onClick={stopCamera}
          >
            Stop Camera
          </button>
        )}

        {/* Capture Image Button */}
        <button
          className="bg-green-500 hover:bg-green-600 px-6 py-3 mt-4 rounded-full text-white transition-all duration-300"
          onClick={captureImage}
        >
          Capture Image
        </button>

        {/* Predict Button (Visible only after capturing) */}
        {capturedImage && (
          <button
            className="bg-rose-500 hover:bg-rose-600 px-6 py-3 mt-4 rounded-full text-white transition-all duration-300"
            onClick={handlePredict}
          >
            Predict
          </button>
        )}

        {/* Display Predicted Emotion */}
        {predictedEmotion && (
          <div className="mt-8 text-2xl text-rose-500 font-normal">
            Decode the Feels: Emotion Unveiled - {predictedEmotion}
          </div>
        )}

        {/* Display Recommended Songs */}
        {recommendedSongs.length > 0 && (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold mb-4 text-blue-500">
              Recommended Songs:
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white border border-gray-300 rounded-lg shadow-md">
                <thead>
                  <tr className="w-full bg-gray-100 border-b">
                    <th className="py-2 px-4 text-left text-gray-600">
                      Song Details
                    </th>
                    <th className="py-2 px-4 text-left text-gray-600">
                      Danceability
                    </th>
                    <th className="py-2 px-4 text-left text-gray-600">
                      Energy
                    </th>
                    <th className="py-2 px-4 text-left text-gray-600">
                      Valence
                    </th>
                    <th className="py-2 px-4 text-left text-gray-600">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {recommendedSongs.map((song, index) => (
                    <tr key={index} className="border-b hover:bg-gray-50">
                      <td className="py-2 px-4">
                        <a
                          href={song.spotify_link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 font-medium"
                        >
                          Listen on Spotify
                        </a>
                      </td>
                      <td className="py-2 px-4 text-gray-700">
                        {song.danceability.toFixed(2)}
                      </td>
                      <td className="py-2 px-4 text-gray-700">
                        {song.energy.toFixed(2)}
                      </td>
                      <td className="py-2 px-4 text-gray-700">
                        {song.valence.toFixed(2)}
                      </td>
                      <td className="py-2 px-4">
                        <a
                          href={song.spotify_link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800"
                        >
                          View on Spotify
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Record;
