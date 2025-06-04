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
    imgsz: int = 960,
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
        patience=50,  # Increased patience
        save=True,
        save_period=10,
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.0005,  # Reduced learning rate
        lrf=0.005,   # Reduced final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,  # Increased warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=5.0,     # Adjusted loss weights
        cls=0.3,
        dfl=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,     # Added rotation
        translate=0.2,    # Increased translation
        scale=0.9,        # Increased scale
        shear=2.0,        # Added shear
        perspective=0.001,# Added perspective
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,        # Added mixup
        copy_paste=0.1,   # Added copy-paste
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
   
    best_model_path = str(data_dir / "training_outputs/train3/weights/best.pt")
    # Train the model
    results = train_model(
        data_yaml_path=data_yaml_path,
        epochs=150,
        batch_size=16,
        imgsz=960,
        device="0",  # Use GPU if available
        pretrained_model=best_model_path,
        project=project_path,
        name="train4"
    )
   
    print("Training completed!")
    print(f"Results saved in: {results.save_dir}")