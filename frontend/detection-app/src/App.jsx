import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState(null);

  const startWebcam = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
      setStream(mediaStream);
      if (webcamRef.current) {
        webcamRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error('Error accessing webcam: ', err);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      if (webcamRef.current) {
        webcamRef.current.srcObject = null;
      }
    }
  }, [stream]);

  const toggleRecording = useCallback(() => {
    if (recording) {
      // Stop recording
      stopWebcam();
      setRecording(false);
    } else {
      // Start recording
      startWebcam();
      setRecording(true);
    }
  }, [recording, startWebcam, stopWebcam]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  return (
    <div className="app-container">
      <h1>Camera Feed</h1>
      
      <div className="webcam-container">
        {/* Video display area - only show when recording */}
        <div className="video-box">
          {recording ? (
            <video 
              ref={webcamRef}
              autoPlay
              playsInline
              muted
            />
          ) : (
            <div className="placeholder-text">
              Press "Start Camera" to begin camera feed
            </div>
          )}
        </div>
        
        {/* Recording controls */}
        <div className="controls">
          <button 
            onClick={toggleRecording} 
            className={recording ? "stop-button" : "start-button"}
          >
            {recording ? "Stop Camera" : "Start Camera"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
