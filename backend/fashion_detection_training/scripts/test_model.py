import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

def load_model(model_path):
    """Load the YOLO model"""
    return YOLO(model_path)

def load_classes(yaml_path):
    """Load class names from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def print_classes(classes):
    """Print all available classes"""
    print("\nAvailable Classes:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

def process_image(model, image_path, output_path, target_class=None, conf_threshold=0.25):
    """Process a single image and save the results"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Perform detection
    results = model(img, conf=conf_threshold)[0]
    
    # Draw bounding boxes
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]
        
        # Skip if we're looking for a specific class and this isn't it
        if target_class is not None and class_name != target_class:
            continue
            
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Processed {image_path} -> {output_path}")

def main():
    # Paths
    model_path = "../data/training_outputs/train3/weights/best.pt"  # Update this to your trained model path
    yaml_path = "../data/data.yaml"
    input_dir = "../data/testing/input"
    output_dir = "../data/testing/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and classes
    model = load_model(model_path)
    classes = load_classes(yaml_path)
    
    # Print all available classes
    print_classes(classes)
    
    '''
    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"detected_{filename}")
            
            # Process image with all classes
            process_image(model, input_path, output_path)
            
            # Example: Process image looking for a specific class (e.g., 'T-shirts_black-white')
            specific_output_path = os.path.join(output_dir, f"specific_T-shirts_black-white_{filename}")
            process_image(model, input_path, specific_output_path, target_class='T-shirts_black-white')
    '''

if __name__ == "__main__":
    main()