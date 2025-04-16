# detection_service/src/yolo_operations.py
import logging
from ultralytics import YOLO
import cv2
import numpy as np

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
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    # detect objects in an image
    def detect_objects_image(self, image_path):
        results = self.model.predict(source=f"{TEST_IMAGE_DIR}/{image_path}", conf=0.25)
        return results
    
    # detect only specified objects in an image
    def detect_objects_image_specified(self, image_path, classes: list[int]):
        results = self.model.predict(source=f"{TEST_IMAGE_DIR}/{image_path}", conf=0.25, classes=classes)
        return results
    
    # detect objects in a video
    def detect_objects_video(self, video_path=None):
        if video_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Live Video', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # detect only specified objects in a video
    def detect_objects_video_specified(self, video_path, classes):
        pass
    
    def display_all_classes(self):
        return self.model.names

def display_boxes_nonspecific():
    yo = YOLOOperations()
    results = yo.detect_objects_image('test_image_street.jpg')
    
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    class_names = result.names
    img = cv2.imread(f"{TEST_IMAGE_DIR}/test_image_street.jpg")
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{class_names[int(class_id)]} {conf:.2f}"
        
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    display_width = 1280
    display_height = 720
    img_display = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('YOLO Object Detection', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_boxes_specified():
    yo = YOLOOperations()
    results = yo.detect_objects_image_specified('test_image_people_car_elephant.jpg', [0])
    
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    class_names = result.names
    img = cv2.imread(f"{TEST_IMAGE_DIR}/test_image_people_car_elephant.jpg")
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{class_names[int(class_id)]} {conf:.2f}"
        
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    display_width = 1280
    display_height = 720
    img_display = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('YOLO Object Detection', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_classes():
    yo = YOLOOperations()
    classes = yo.display_all_classes()
    for i in range(len(classes)):
        print(f"{i}: {classes[i]}")

def detect_objects_video():
    yo = YOLOOperations()
    yo.detect_objects_video()

def main():
    #display_boxes_nonspecific()
    #show_classes()
    detect_objects_video()
    
if __name__ == "__main__":
    print("Running YOLO operations")
    main()