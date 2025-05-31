from ultralytics import YOLO
import os
import yaml
import torch
from pathlib import Path

print(torch.cuda.is_available())
print(torch.cuda.device_count())

def train_model(
    data_yaml_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    pretrained_model: str = "yolov8s.pt",
    project: str = "fashion_detection",
    name: str = "train1"
):
    """
    Train a YOLOv8 model on the fashion dataset.
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Input image size
        device: Device to use for training (GPU ID or 'cpu')
        pretrained_model: Path to pretrained model weights
        project: Project name for saving results
        name: Run name for saving results
    """
    # Load the model
    model = YOLO(pretrained_model)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=30,  # Increased patience
        save=True,
        save_period=10,
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",  # Explicitly set optimizer
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # L2 regularization
        warmup_epochs=3,  # Learning rate warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution focal loss gain
        hsv_h=0.015,  # HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,  # Flip up-down augmentation
        fliplr=0.5,  # Flip left-right augmentation
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.0,  # Mixup augmentation
        copy_paste=0.0,  # Copy-paste augmentation
        verbose=True,
        seed=42,
    )
    
    return results

if __name__ == "__main__":
    # Get the absolute path to the data.yaml file and project directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    data_yaml_path = str(data_dir / "data.yaml")
    
    # Set project path to be inside the data directory
    project_path = str(data_dir / "training_outputs")
    
    best_model_path = str(data_dir / "training_outputs/train1/weights/best.pt")
    # Train the model
    results = train_model(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        imgsz=640,
        device="0",  # Use GPU if available
        pretrained_model=best_model_path,
        project=project_path,
        name="train3"
    )
    
    print("Training completed!")
    print(f"Results saved in: {results.save_dir}") 