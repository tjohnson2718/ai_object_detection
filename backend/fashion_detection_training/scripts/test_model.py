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
 
    # Store original image dimensions
    original_height, original_width = img.shape[:2]
   
    # Resize image to match training size while maintaining aspect ratio
    target_size = 960
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
   
    # Resize image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
   
    # Create a square canvas of target size
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
   
    # Place resized image in the center of the canvas
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
 
    # Perform detection on the preprocessed image
    results = model(canvas, conf=conf_threshold, iou=0.45)[0]
   
    # Create a copy of the original image for drawing
    output_img = img.copy()
   
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
       
        # Convert coordinates back to original image size
        x1 = int((x1 - x_offset) / scale)
        y1 = int((y1 - y_offset) / scale)
        x2 = int((x2 - x_offset) / scale)
        y2 = int((y2 - y_offset) / scale)
       
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
           
        # Draw rectangle
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        # Add label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)
    print(f"Processed {image_path} -> {output_path}")
 
def main():
    # Paths
    model_path = "../data/fashion_models/training_outputs/train3/weights/best.pt"
    yaml_path = "../data/fashion_models/data.yaml"
    input_dir = "../data/testing/input"
    output_dir = "../data/testing/output"
   
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
   
    # Load model and classes
    model = load_model(model_path)
    classes = load_classes(yaml_path)
   
    # Print all available classes
    print_classes(classes)
   
 
    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"detected_{filename}")
           
            # Process image with all classes
            process_image(model, input_path, output_path, conf_threshold=0.4)
           
            # Example: Process image looking for a specific class (e.g., 'T-shirts_black-white')
            #specific_output_path = os.path.join(output_dir, f"specific_T-shirts_black-white_{filename}")
            #process_image(model, input_path, specific_output_path, target_class='T-shirts_black-white')
   
 
if __name__ == "__main__":
    main()