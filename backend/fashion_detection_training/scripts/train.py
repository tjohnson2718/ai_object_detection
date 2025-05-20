from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def train_model(
    data_yaml_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    pretrained_model: str = "yolov8n.pt",
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
        patience=20,  # Early stopping patience
        save=True,  # Save best model
        save_period=10,  # Save checkpoint every 10 epochs
        exist_ok=True,  # Overwrite existing experiment
        pretrained=True,  # Use pretrained weights
        optimizer="auto",  # Auto-select optimizer
        verbose=True,  # Print training progress
        seed=42,  # For reproducibility
    )
    
    return results

if __name__ == "__main__":
    # Get the absolute path to the data.yaml file and project directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    data_yaml_path = str(data_dir / "data.yaml")
    
    # Set project path to be inside the data directory
    project_path = str(data_dir / "training_outputs")
    
    # Train the model
    results = train_model(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        imgsz=640,
        device="0",  # Use GPU if available
        pretrained_model="yolov8n.pt",
        project=project_path,
        name="train1"
    )
    
    print("Training completed!")
    print(f"Results saved in: {results.save_dir}") 