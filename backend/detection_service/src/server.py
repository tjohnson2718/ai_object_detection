# detection_service/src/server.py
import base64
import io
import logging
import os
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
from dotenv import load_dotenv
import torch

from yolo_service import YOLOOperations
from language_service import LanguageService

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(name="Object Detection Service")

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model with GPU support
model_path = os.environ.get("MODEL_PATH", "./yolov8n.pt")
conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", '0.5'))
iou_threshold = float(os.environ.get("IOU_THRESHOLD", "0.45"))

# Initialize YOLO with GPU support
yolo_ops = YOLOOperations(
    model_path=model_path,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold
)

if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch CUDA: {torch.version.cuda}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("No GPU detected. Running on CPU will be significantly slower.")

logger.info(f"Detection service initialized with model: {model_path}")

class DetectionRequest(BaseModel):
    classes: Optional[List[str]] = None
    image_data: str
    timestamp: int = 0
    
class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox
    
class DetectionResponse(BaseModel):
    detections: List[Detection]
    timestamp: int
    frame_id: int = 0
    
class BatchDetectionRequest(BaseModel):
    requests: List[DetectionRequest]

class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    
class QueryRequest(BaseModel):
    query: str

@app.post("/parse_query")
async def parse_query(request: QueryRequest):
    try:
        logger.info(f"Parsing query: {request.query}")
        language_service = LanguageService()
        classes = language_service.parse_query(request.query)
        logger.info(f"Parsed classes: {classes}")
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error in parse_query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest,
                         run_custom_detection: bool = False,
                         custom_classes: list[str] = []):
    try:
        # Use the classes directly from the request
        classes = request.classes
        logger.info(f"Detection classes: {classes}")
        try:
            if "," in request.image_data:
                base64_data = request.image_data.split(",")[1]
            else:
                base64_data = request.image_data
            
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data format")
        
        # First pass: Yolo detection
        detections = yolo_ops.detect_objects(image_bytes, classes)
        logger.info(f"YOLO detection returned {len(detections)} objects")
        
        # Format the response
        detection_objects = []
        for det in detections:
            detection_objects.append(Detection(
                class_name=det["class"],
                confidence=det["confidence"],
                bbox=BoundingBox(
                    x_min=det["bbox"][0],
                    y_min=det["bbox"][1],
                    x_max=det["bbox"][2],
                    y_max=det["bbox"][3]
                )
            ))
        
        logger.info(f"Formatted {len(detection_objects)} detections for response")
        
        return DetectionResponse(
            detections=detection_objects,
            timestamp=request.timestamp,
            frame_id=0
        )
        
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/classes")
async def get_classes():
    try:
        classes = yolo_ops.get_classes()
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error in get_classes: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2)
        }
    
    return {
        "status": "healthy", 
        "model": model_path,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }

@app.post("/batch_detect", response_model=BatchDetectionResponse)
async def batch_detect_objects(batch_request: BatchDetectionRequest,
                             run_custom_detection: bool = False,
                             custom_classes: list[str] = []):
    try:
        # Extract batch of images and timestamps
        batch_images = []
        timestamps = []
        
        for req in batch_request.requests:
            try:
                if "," in req.image_data:
                    base64_data = req.image_data.split(",")[1]
                else:
                    base64_data = req.image_data
                
                image_bytes = base64.b64decode(base64_data)
                batch_images.append(image_bytes)
                timestamps.append(req.timestamp)
            except Exception as e:
                logger.error(f"Error decoding base64 image: {e}")
                raise HTTPException(status_code=400, detail="Invalid image data format")
        
        # Process batch with GPU acceleration
        batch_results = yolo_ops.batch_detect_objects(batch_images)
        
        # Format response
        results = []
        for idx, detections in enumerate(batch_results):
            detection_objects = []
            for det in detections:
                detection_objects.append(Detection(
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox=BoundingBox(
                        x_min=det["bbox"]["x_min"],
                        y_min=det["bbox"]["y_min"],
                        x_max=det["bbox"]["x_max"],
                        y_max=det["bbox"]["y_max"]
                    )
                ))
            
            results.append(DetectionResponse(
                detections=detection_objects,
                timestamp=timestamps[idx],
                frame_id=idx
            ))
        
        return BatchDetectionResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error in batch_detect_objects: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)



