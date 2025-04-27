export const getBase64FromVideo = async (videoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.9);
};

export const detectObjectsInFrame = async (base64Image) => {
    try {
        // Log the first 50 characters to check format
        console.log("Image data preview:", base64Image.substring(0, 50));
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                request: {
                    image_data: base64Image,
                    timestamp: Date.now()
                }
            }),
        });

        // Log the response status and any error text
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

export const detectSpecificObjects = async (base64Image, classNames) => {
    try {
        const queryParams = classNames.map(cls => `custom_classes=${encodeURIComponent(cls)}`).join('&');
        const url = `/api/detect?${queryParams}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                request: {
                    image_data: base64Image,
                    timestamp: Date.now()
                }
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error in specific object detection: ', error);
        throw error;
    }
};