# detection_service/src/yolo_operations.py
import torch
import time
import logging
import os
import io
from typing import List, Dict, Any, Union, Optional

import cv2
import numpy as np
from PIL import Image
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

class YOLOOperations:
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLOOperations with model path and detection thresholds.
        
        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
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
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
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
    
    def detect_objects(self, 
                      image_data: bytes, 
                      classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in an image with GPU acceleration.
        
        Args:
            image_data: Raw bytes of the image (JPEG/PNG)
            classes: Optional list of class IDs to filter (None means detect all)
            
        Returns:
            List of detections with class, confidence, and normalized bounding box
        """
        try:
            start_time = time.time()
            
            img = self._bytes_to_image(image_data)
            height, width = img.shape[:2]
            
            results = self.model.predict(
                source=img, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes,
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
                            "class": result.names[int(class_id)],
                            "confidence": float(conf),
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm]
                        })
            
            inference_time = time.time() - start_time
            logger.info(f"Detection completed in {inference_time:.4f}s on {self.device}")
            logger.info(f"Detected {len(detections)} objects")
            
            return detections
        
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def detect_objects_by_names(self, 
                              image_data: bytes, 
                              class_names: List[str]) -> List[Dict[str, Any]]:
        """
        Detect objects in an image by class names.
        
        Args:
            image_data: Raw bytes of the image
            class_names: List of class names to detect
            
        Returns:
            List of detections for the specified classes
        """
        try:
            class_ids = []
            for name in class_names:
                for id, model_name in self.model.names.items():
                    if model_name.lower() == name.lower():
                        class_ids.append(id)
                        break
            
            if not class_ids:
                logger.warning(f"None of the requested classes {class_names} were found in the model")
                return []
            
            return self.detect_objects(image_data, class_ids)
            
        except Exception as e:
            logger.error(f"Error detecting objects by names: {e}")
            return []

    def class_name_exists(self, class_name: str) -> bool:
        """
        Check if a class name exists in the model.
        
        Args:
            class_name: Name of the class
            
        Returns:
            True if the class exists, False otherwise
        """
        try:
            for name in self.model.names.values():
                if name.lower() == class_name.lower():
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking class name: {e}")
            return False
    
    def detect_objects_filtered(self, 
                              image_data: bytes, 
                              class_names: List[str]) -> List[Dict[str, Any]]:
        """
        Detect only specific objects in an image by class name.
        
        Args:
            image_data: Raw bytes of the image
            class_names: List of class names to detect
            
        Returns:
            List of filtered detections
        """
        try:
            all_detections = self.detect_objects(image_data)
            
            filtered_detections = [
                det for det in all_detections 
                if det["class_name"].lower() in [name.lower() for name in class_names]
            ]
            
            logger.info(f"Filtered to {len(filtered_detections)} objects of classes: {class_names}")
            return filtered_detections
        
        except Exception as e:
            logger.error(f"Error filtering detections: {e}")
            return []
    
    def create_annotated_image(self, 
                              image_data: bytes, 
                              detections: List[Dict[str, Any]]) -> bytes:
        """
        Create an annotated image with bounding boxes and labels.
        
        Args:
            image_data: Raw bytes of the image
            detections: List of detections to visualize
            
        Returns:
            JPEG encoded bytes of the annotated image
        """
        try:
            img = self._bytes_to_image(image_data)
            
            for det in detections:
                if "bbox_pixels" in det:
                    x1, y1, x2, y2 = det["bbox_pixels"]
                else:
                    height, width = img.shape[:2]
                    x1, y1, x2, y2 = det["bbox"]["x_min"], det["bbox"]["y_min"], det["bbox"]["x_max"], det["bbox"]["y_max"]
                    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{det['class_name']} {det['confidence']:.2f}"
                
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
        
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return b''
    
    def process_video_frame(self, 
                           frame: np.ndarray, 
                           classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Process a video frame and detect objects.
        
        Args:
            frame: NumPy array of the video frame
            classes: Optional list of class IDs to filter
            
        Returns:
            List of detections
        """
        try:
            height, width = frame.shape[:2]
            
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes
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
                            "class": result.names[int(class_id)],
                            "confidence": float(conf),
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm]
                        })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return []
            
    def get_class_id_by_name(self, class_name: str) -> Optional[int]:
        """
        Get class ID by name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Class ID or None if not found
        """
        try:
            for class_id, name in self.model.names.items():
                if name.lower() == class_name.lower():
                    return class_id
            return None
        except Exception as e:
            logger.error(f"Error getting class ID: {e}")
            return None
    
    def get_class_names_by_ids(self, class_ids: List[int]) -> List[str]:
        """
        Get class names by IDs.
        
        Args:
            class_ids: List of class IDs
            
        Returns:
            List of class names
        """
        try:
            return [self.model.names.get(class_id, "unknown") for class_id in class_ids]
        except Exception as e:
            logger.error(f"Error getting class names: {e}")
            return []
    
    def batch_detect_objects(self, 
                           batch_image_data: List[bytes], 
                           classes: Optional[List[int]] = None) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of images at once for improved throughput.
        
        Args:
            batch_image_data: List of raw image bytes
            classes: Optional list of class IDs to filter
            
        Returns:
            List of detection lists, one for each input image
        """
        try:
            start_time = time.time()
            
            # Convert all images to numpy arrays
            images = []
            dimensions = []
            
            for img_data in batch_image_data:
                img = self._bytes_to_image(img_data)
                images.append(img)
                dimensions.append((img.shape[0], img.shape[1]))  # height, width
            
            # Run batch inference
            results = self.model.predict(
                source=images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes,
                device=self.device  # Use GPU if available
            )
            
            # Process results for each image
            all_detections = []
            
            for i, result in enumerate(results):
                height, width = dimensions[i]
                detections = []
                
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy'):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box
                        
                        # Normalize coordinates
                        x1_norm = float(x1 / width)
                        y1_norm = float(y1 / height)
                        x2_norm = float(x2 / width)
                        y2_norm = float(y2 / height)
                        
                        detections.append({
                            "class": result.names[int(class_id)],
                            "confidence": float(conf),
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm]
                        })
                
                all_detections.append(detections)
            
            # Log batch processing performance
            batch_time = time.time() - start_time
            avg_time = batch_time / len(batch_image_data) if batch_image_data else 0
            logger.info(f"Batch of {len(batch_image_data)} images processed in {batch_time:.4f}s " 
                       f"({avg_time:.4f}s per image) on {self.device}")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Error in batch detection: {e}")
            return [[] for _ in range(len(batch_image_data))]
    