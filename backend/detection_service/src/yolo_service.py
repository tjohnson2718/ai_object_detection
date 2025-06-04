# detection_service/src/yolo_operations.py
import torch
import time
import logging
import os
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Union, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_IMAGE_DIR = 'detection_service/data'
""" YOLO classes:
    0: person
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
    5: bus
    6: train
    7: truck
    8: boat
    9: traffic light
    10: fire hydrant
    11: stop sign
    12: parking meter
    13: bench
    14: bird
    15: cat
    16: dog
    17: horse
    18: sheep
    19: cow
    20: elephant
    21: bear
    22: zebra
    23: giraffe
    24: backpack
    25: umbrella
    26: handbag
    27: tie
    28: suitcase
    29: frisbee
    30: skis
    31: snowboard
    32: sports ball
    33: kite
    34: baseball bat
    35: baseball glove
    36: skateboard
    37: surfboard
    38: tennis racket
    39: bottle
    40: wine glass
    41: cup
    42: fork
    43: knife
    44: spoon
    45: bowl
    46: banana
    47: apple
    48: sandwich
    49: orange
    50: broccoli
    51: carrot
    52: hot dog
    53: pizza
    54: donut
    55: cake
    56: chair
    57: couch
    58: potted plant
    59: bed
    60: dining table
    61: toilet
    62: tv
    63: laptop
    64: mouse
    65: remote
    66: keyboard
    67: cell phone
    68: microwave
    69: oven
    70: toaster
    71: sink
    72: refrigerator
    73: book
    74: clock
    75: vase
    76: scissors
    77: teddy bear
    78: hair drier
    79: toothbrush
"""

class YOLOService:
    _instance = None
    _initialized = False
    _thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(YOLOService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 model_path: str = 'models/yolov8n.pt',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLOService with model path and detection thresholds.
        
        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        if YOLOService._initialized:
            return
        
        try:
            # Check for CUDA availability
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load the model
            self.model = YOLO(model_path)
            
            # Move model to the right device
            if self.device.type == 'cuda':
                self.model.to(self.device)
                logger.info(f"Model successfully loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA is not available, using CPU for inference. This will be slower.")
                
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            logger.info(f"YOLO model loaded from {model_path}")
            logger.info(f"Using confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
            
            YOLOService._initialized = True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def run_inference(self, image_data: bytes, classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for object detection.
        
        Args:
            image_data: Raw bytes of the image (JPEG/PNG)
            classes: Optional list of class names to filter detections
            
        Returns:
            List of detections with class, confidence, and normalized bounding box
        """
        try:
            start_time = time.time()
            
            # Image bytes to numpy array
            img = self._bytes_to_image(image_data)
            height, width = img.shape[:2]
            
            # Initialize class_ids as None
            class_ids = None
            
            # Convert class names to IDs if provided
            if classes and len(classes) > 0:
                class_ids = self._get_class_ids(classes)
                
            # Run Inference
            results = self.model.predict(
                source=img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=class_ids,  # This will be None if no classes were provided
                device=self.device
            )
            
            detections = []
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy'):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box
                    
                        x1_norm = float(x1 / width)
                        y1_norm = float(y1 / height)
                        x2_norm = float(x2 / width)
                        y2_norm = float(y2 / height)
                        
                        detections.append({
                            "class": self.model.names[int(class_id)],
                            "confidence": float(conf),
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm]
                        })

            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            logger.info(f"Detected {len(detections)} objects")
            
            return detections
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            return []
      
    async def arun_inference(self, image_data: bytes, classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Asynchronous entry point for object detection.
        Runs inference in a separate thread to avoid blocking the event loop.
        
        Args:
            image_data: Raw bytes of the image (JPEG/PNG)
            classes: Optional list of class names to filter (None means detect all)
            
        Returns:
            List of detections with class, confidence, and normalized bounding box
        """
        try:
            # Run the synchronous inference in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._thread_pool,
                self.run_inference,
                image_data,
                classes
            )
            return result
        except Exception as e:
            logger.error(f"Error in async inference: {e}")
            return []
        
    def get_classes(self) -> List[str]:
        """
        Get list of classes the model can detect.
        
        Returns:
            List of class names
        """
        try:
            return list(self.model.names.values())
        except Exception as e:
            logger.error(f"Error getting classes: {e}")
            return []

    def _bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array.
        
        Args:
            image_bytes: Raw bytes of the image (JPEG/PNG)
            
        Returns:
            Numpy array representation of the image
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image bytes")
                raise ValueError("Invalid image data")
                
            return img
        except Exception as e:
            logger.error(f"Error converting bytes to image: {e}")
            raise ValueError(f"Error processing image: {e}")
    
    def _get_class_ids(self, classes: List[str]) -> List[int]:
        """
        Get class IDs from provided class names.
        
        Args:
            classes: List of class names
        
        Returns:
            List of class IDs
        """
        try:
            if classes is None:
                return None
            
            class_ids = []
            for name in classes:
                for id, model_name in self.model.names.items():
                    if model_name.lower() == name.lower():
                        class_ids.append(id)
                        break
            return class_ids
        except Exception as e:
            logger.error(f"Error getting class IDs: {e}")
            return []
        
            
            