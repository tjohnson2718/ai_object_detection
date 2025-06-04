const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const getBase64FromVideo = async (videoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.9);
};

export const detectObjectsInFrame = async (base64Image, yoloClasses = null, clothingClasses = null) => {
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_data: base64Image,
                timestamp: Date.now(),
                yolo_classes: yoloClasses,
                clothing_classes: clothingClasses
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Error ${response.status}: ${errorText}`);
            throw new Error(`HTTP error! Status: ${response.status}, Details: ${errorText}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error in object detection:', error);
        throw error;
    }
};
