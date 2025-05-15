# backend/detection_service/src/custom_service.py

from ultralytics import YOLO
import torch

def setup_fashion_model():
    print("Downloading fashion detection model...")
    try:
        model = YOLO("keremberke/yolov8n-fashion-detection")
        
        print("\nAvailable classes:")
        
        for class_id, class_name in model.names.items():
            print(f"{class_id}: {class_name}")
            
        return model
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None
    
    
if __name__ == "__main__":
    model = setup_fashion_model()

