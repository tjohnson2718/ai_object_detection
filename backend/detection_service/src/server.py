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

from yolo_service import YOLOService
from clothing_service import ClothingService
from language_service import LanguageService
from routing.language import language_router

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

app.include_router(language_router)

# Initialize model with GPU support
model_path = os.environ.get("MODEL_PATH", "./yolov8n.pt")
conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", '0.5'))
iou_threshold = float(os.environ.get("IOU_THRESHOLD", "0.45"))

# Initialize Detection Services with GPU support
yolo_service = YOLOService(
    model_path=model_path,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold
)

clothing_service = ClothingService(
    model_path='models/best.pt',
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

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class Detection(BaseModel):
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    
class DetectionRequest(BaseModel):
    image_data: str
    timestamp: int = 0
    yolo_classes: Optional[List[str]] = None
    clothing_classes: Optional[List[str]] = None
    
class DetectionResponse(BaseModel):
    detections: list[Detection]
    timestamp: int = 0
    frame_id: int = 0
    
@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    try:
        try:
            if "," in request.image_data:
                base64_data = request.image_data.split(",")[1]
            else:
                base64_data = request.image_data
                
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        detections = []
        if request.yolo_classes and not request.clothing_classes: # YOLO only
            detections.extend(await yolo_service.arun_inference(image_bytes, request.yolo_classes))
        elif request.clothing_classes and not request.yolo_classes: # Clothing only
            detections.extend(await clothing_service.arun_inference(image_bytes, request.clothing_classes))
        else: # YOLO and Clothing
            yolo_detections = await yolo_service.arun_inference(image_bytes, request.yolo_classes)
            clothing_detections = await clothing_service.arun_inference(image_bytes, request.clothing_classes)
            detections.extend(yolo_detections)
            detections.extend(clothing_detections)
        
        logger.info(f"Detected {len(detections)} objects")
        
        # Format the response
        detection_objects = []
        for det in detections:
            detection_objects.append(Detection(
                class_name=det["class"],
                confidence=det["confidence"],
                bounding_box=BoundingBox(
                    x_min=det["bbox"][0],
                    y_min=det["bbox"][1],
                    x_max=det["bbox"][2],
                    y_max=det["bbox"][3]
                )
            ))
            
        return DetectionResponse(
            detections=detection_objects,
            timestamp=request.timestamp,
            frame_id=0
        )
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    try:
        classes = yolo_service.get_classes()
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
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
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)



