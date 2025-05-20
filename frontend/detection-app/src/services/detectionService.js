export const getBase64FromVideo = async (videoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.9);
};

export const parseQuery = async (query) => {
    try {
        console.log("Sending query for parsing:", query);
        const response = await fetch('/api/parse_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Error ${response.status}: ${errorText}`);
            throw new Error(`HTTP error! Status: ${response.status}, Details: ${errorText}`);
        }

        const result = await response.json();
        console.log("Parsed query result:", result);
        return result.classes;
    } catch (error) {
        console.error('Error parsing query:', error);
        throw error;
    }
};

export const detectObjectsInFrame = async (base64Image, classes = null) => {
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                request: {
                    image_data: base64Image,
                    timestamp: Date.now(),
                    classes: classes  // Now we pass the classes directly
                }
            }),
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