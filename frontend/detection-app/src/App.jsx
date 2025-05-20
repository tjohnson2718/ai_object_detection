import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';
import { getBase64FromVideo, detectObjectsInFrame, parseQuery } from './services/detectionService';
import DetectionOverlay from './components/DetectionOverlay';

function App() {
  const webcamRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0});
  const [query, setQuery] = useState('');
  const [detectionClasses, setDetectionClasses] = useState(null);

  const detectionIntervalRef = useRef(null);

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

    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    setDetections([]);
    setIsDetecting(false);
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

  const handleVideoPlay = () => {
    if (webcamRef.current) {
      setVideoSize({
        width: webcamRef.current.videoWidth,
        height: webcamRef.current.videoHeight
      });
    }
  };

  const toggleDetection = useCallback(() => {
    setIsDetecting(prevState => {
      if (!prevState) {
        // Start detection with all classes (null)
        setDetectionClasses(null);
        runDetection();

        detectionIntervalRef.current = setInterval(() => {
          runDetection();
        }, 200);

        return true;
      } else {
        // Stop detection
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
          detectionIntervalRef.current = null;
        }
        return false;
      }
    });
  }, []);

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    
    // Stop current detection if running
    if (isDetecting) {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      setIsDetecting(false);
    }

    try {
      // Parse the query to get classes
      const classes = await parseQuery(query);
      setDetectionClasses(classes);
      
      // Start new detection with the parsed classes
      setTimeout(() => {
        setIsDetecting(true);
        runDetection();
        detectionIntervalRef.current = setInterval(() => {
          runDetection();
        }, 200);
      }, 100);
    } catch (error) {
      console.error('Error processing query:', error);
      // Optionally show an error message to the user
    }
  };

  const runDetection = async () => {
    if (!webcamRef.current || !webcamRef.current.videoWidth) return;
    
    try {
      const base64Image = await getBase64FromVideo(webcamRef.current);
      
      setVideoSize({
        width: webcamRef.current.videoWidth,
        height: webcamRef.current.videoHeight
      });
      
      // Pass the current detection classes to the detection service
      const result = await detectObjectsInFrame(base64Image, detectionClasses);
      
      if (result && result.detections && Array.isArray(result.detections)) {
        const validDetections = result.detections.filter(det => 
          det && det.bbox && 
          typeof det.bbox.x_min === 'number' && 
          typeof det.bbox.y_min === 'number' && 
          typeof det.bbox.x_max === 'number' && 
          typeof det.bbox.y_max === 'number'
        );
        
        setDetections(validDetections);
      } else {
        console.error("Unexpected response format:", result);
        setDetections([]);
      }
    } catch (error) {
      console.error('Detection error:', error);
      setDetections([]);
    }
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, [stream]);

  return (
    <div className="app-container">
      <h1>Object Detection</h1>
      
      <div className="webcam-container">
        <div className="video-box">
          {recording ? (
            <div className="video-with-overlay">
              <video 
                ref={webcamRef}
                autoPlay
                playsInline
                muted
                onPlay={handleVideoPlay}
              />
              {isDetecting && (
                <DetectionOverlay 
                  detections={detections}
                  videoWidth={videoSize.width}
                  videoHeight={videoSize.height}
                />
              )}
            </div>
          ) : (
            <div className="placeholder-text">
              Press "Start Camera" to begin camera feed
            </div>
          )}
        </div>
        
        {recording && (
          <form onSubmit={handleQuerySubmit} className="query-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter what you want to detect (e.g., 'Show me all cars and people')"
              className="query-input"
            />
            <button 
              type="submit" 
              className="query-submit"
              disabled={!query.trim()} // Disable if query is empty
            >
              Apply Query
            </button>
          </form>
        )}
        
        <div className="controls">
          <button 
            onClick={toggleRecording} 
            className={recording ? "stop-button" : "start-button"}
          >
            {recording ? "Stop Camera" : "Start Camera"}
          </button>
          
          {recording && (
            <button
              onClick={toggleDetection}
              className={isDetecting ? "stop-detection-button" : "start-detection-button"}
              disabled={!recording}
            >
              {isDetecting ? "Stop Detection" : "Start Detection"}
            </button>
          )}
        </div>
        
        {isDetecting && detections.length > 0 && (
          <div className="detection-stats">
            <h3>Detected Objects: {detections.length}</h3>
            <ul>
              {detections.map((det, index) => (
                <li key={index}>
                  {det.class_name}: {(det.confidence * 100).toFixed(1)}%
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
