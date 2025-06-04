import { useState, useRef, useCallback, useEffect } from 'react';
import { getBase64FromVideo, detectObjectsInFrame } from './services/detectionService';
import DetectionOverlay from './components/DetectionOverlay';
import './App.css';
import { QueryParser } from './components/QueryParser';

function App() {
  const webcamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const currentClassesRef = useRef({ yolo: [], clothing: [] });
  
  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0});
  const [detectionStats, setDetectionStats] = useState({
    requestsSent: 0,
    responsesReceived: 0
  });

  const startWebcam = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 1280,
          height: 720
        },
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
      console.log(webcamRef.current.videoWidth, webcamRef.current.videoHeight);
    }
  };

  const toggleDetection = useCallback(() => {
    setIsDetecting(prevState => {
      if (!prevState) {
        // Reset counters when starting detection
        setDetectionStats({
          requestsSent: 0,
          responsesReceived: 0
        });
        
        runDetection();

        detectionIntervalRef.current = setInterval(() => {
          runDetection();
        }, 250);

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

  const handleQuerySubmit = async (result) => {
    console.log('Query submit result:', result);
    
    // Stop current detection if running
    if (isDetecting) {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      setIsDetecting(false);
    }

    // Update the classes in the ref
    currentClassesRef.current = {
      yolo: result.yolo_classes || [],
      clothing: result.clothing_classes || []
    }
    
    console.log('Current classes:', currentClassesRef.current);
    
    // Start new detection with the parsed classes
    setIsDetecting(true);
    runDetection();
    detectionIntervalRef.current = setInterval(() => {
      runDetection();
    }, 250);
  };

  const runDetection = async () => {
    if (!webcamRef.current || !webcamRef.current.videoWidth) return;
    
    try {
      const base64Image = await getBase64FromVideo(webcamRef.current);
      
      console.log('Running detection with classes:', currentClassesRef.current);

      setVideoSize({
        width: webcamRef.current.videoWidth,
        height: webcamRef.current.videoHeight
      });
      
      setDetectionStats(prev => ({
        ...prev,
        requestsSent: prev.requestsSent + 1
      }));
      
      const result = await detectObjectsInFrame(base64Image, currentClassesRef.current.yolo, currentClassesRef.current.clothing);
      
      if (result && result.detections && Array.isArray(result.detections)) {
        const validDetections = result.detections.filter(det => 
          det && det.bounding_box && 
          typeof det.bounding_box.x_min === 'number' && 
          typeof det.bounding_box.y_min === 'number' && 
          typeof det.bounding_box.x_max === 'number' && 
          typeof det.bounding_box.y_max === 'number'
        );
        
        setDetections(validDetections);
        
        setDetectionStats(prev => ({
          ...prev,
          responsesReceived: prev.responsesReceived + 1
        }));
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
          <QueryParser onSubmit={handleQuerySubmit} />
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
        
        {isDetecting && (
          <div className="detection-stats">
            <h3>Detection Statistics</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Requests Sent:</span>
                <span className="stat-value">{detectionStats.requestsSent}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Responses Received:</span>
                <span className="stat-value">{detectionStats.responsesReceived}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Pending:</span>
                <span className="stat-value">
                  {detectionStats.requestsSent - detectionStats.responsesReceived}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
