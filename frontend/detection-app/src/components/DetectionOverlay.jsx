import { useRef, useEffect } from 'react';

const DetectionOverlay = ({ detections, videoWidth, videoHeight }) => {
    const canvasRef = useRef(null);

    const colors = {
        person: 'rgba(255, 0, 0, 0.7)',   // red
        car: 'rgba(0, 255, 0, 0.7)',      // green
        dog: 'rgba(0, 0, 255, 0.7)',      // blue
        laptop: 'rgba(0, 255, 255, 0.7)', // cyan
        default: 'rgba(255, 255, 0, 0.7)' // yellow
      };

      useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        canvas.width = videoWidth;
        canvas.height = videoHeight;

        if (!detections || !Array.isArray(detections) || detections.length === 0) {
            console.log("No valid detections to render", detections);
            return;
        }

        console.log("Drawing detections:", detections);

        detections.forEach(detection => {
            if (!detection || !detection.bbox) {
                console.warn("Invalid detection format:", detection);
                return;
            }

            const bbox = detection.bbox;

            const x1 = bbox.x_min;
            const y1 = bbox.y_min;
            const x2 = bbox.x_max;
            const y2 = bbox.y_max;

            const boxX = x1 * videoWidth;
            const boxY = y1 * videoHeight;
            const boxWidth = (x2 - x1) * videoWidth;
            const boxHeight = (y2 - y1) * videoHeight;

            const className = detection.class_name || 'unknown';
            const color = colors[className.toLowerCase()] || colors.default;

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

            const confidence = detection.confidence || 0;
            const label = `${className} ${(confidence * 100).toFixed(1)}%`;
            ctx.fillStyle = color;
            ctx.font = '18px Arial';
            const textWidth = ctx.measureText(label).width;

            ctx.fillRect(boxX, boxY - 25, textWidth + 10, 25);

            ctx.fillStyle = 'white';
            ctx.fillText(label, boxX + 5, boxY - 7);
        });
      }, [detections, videoWidth, videoHeight, colors]);

      return (
        <canvas
            ref={canvasRef}
            className="detection-overlay"
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
            }}
        />
      );
};

export default DetectionOverlay;