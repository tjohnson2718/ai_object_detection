import logging
import os
from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew
from textwrap import dedent
from dotenv import load_dotenv
from pydantic import BaseModel
import json

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try multiple ways to get the API key
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or  # Try environment variable
    os.getenv("OPENAI_KEY") or      # Try alternative name
    os.environ.get("OPENAI_API_KEY") # Try direct environ access
)

if not OPENAI_API_KEY:
    logger.error("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or add it to .env file")
    raise ValueError("OpenAI API key is required")

logger.info("OpenAI API key found and loaded successfully")

class AnalyzedQueryOutput(BaseModel):
    query: str
    yolo_classes: Optional[List[str]] = None
    clothing_classes: Optional[List[str]] = None

class LanguageService():
    def __init__(self):
        self.yolo_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self.clothing_classes = [
            '-', 'Blouse_black-white', 'Blouse_deep', 'Blouse_pastel', 'Cardigan_black-white', 'Cardigan_deep', 
            'Cardigan_pastel', 'Cardigan_pastern', 'Cardigan_pattern', 'Cardigan_vivid', 'Denim-pants_black-white', 
            'Denim-pants_deep', 'Denim-pants_pastel', 'Denim-pants_pattern', 'Denim-pants_vivid', 'Hoodie_black-white', 
            'Hoodie_deep', 'Hoodie_pastel', 'Hoodie_vivid', 'Long-skirt_black-white', 'Long-skirt_deep', 
            'Long-sleeves_black-white', 'Long-sleeves_deep', 'Long-sleeves_pastel', 'Long-sleeves_pattern', 
            'Long-sleeves_vivid', 'Midi-skirts_black-white', 'Midi-skirts_deep', 'Midi-skirts_pastel', 'Midi-skirts_vivid', 
            'Mini-skirts_deep', 'One-piece_black-white', 'One-piece_deep', 'One-piece_pastel', 'Pk-shirts_black-white', 
            'Pk-shirts_deep', 'Pk-shirts_pastel', 'Pk-shirts_pattern', 'Pk-shirts_vivid', 'Shirts-pastel', 'Shirts_black-white', 
            'Shirts_deep', 'Shirts_pastel', 'Shirts_pattern', 'Shirts_vivid', 'Short-skirt_black-white', 'Short-skirt_deep', 
            'Short-skirt_pattern', 'Short-sleeves_black-white', 'Short-sleeves_deep', 'Short-sleeves_pastel', 
            'Short-sleeves_pattern', 'Short-sleeves_vivid', 'Shorts_black-white', 'Shorts_deep', 'Shorts_pastel', 'Shorts_pattern', 
            'Shorts_vivid', 'Slacks_black-white', 'Slacks_deep', 'Slacks_pastel', 'Slacks_pattern', 'Slacks_vivid', 
            'Sleeveless_black-white', 'Sleeveless_deep', 'Sleeveless_pastel', 'Sleeveless_pattern', 'Slim-pants_black-white', 
            'Slim-pants_deep', 'Slim-pants_pastel', 'Slim-pants_pattern', 'Slim-pants_vivid', 'Straight-pants_black-white', 
            'Straight-pants_deep', 'Straight-pants_pastel', 'Straight-pants_vivid', 'Sweatpants_black-white', 'Sweatshirt_black-white', 
            'Sweatshirt_deep', 'Sweatshirt_pastel', 'Sweatshirt_pattern', 'Sweatshirt_vivid', 'T-shirts_black-white', 
            'Training-pants_black-white', 'Training-pants_deep', 'Training-pants_pastel', 'Training-pants_pattern', 'Training-pants_vivid']
        
        # Create the query analyzer agent
        self.query_analyzer = Agent(
            role='Query Analyzer',
            goal='Analyze user queries and identify relevant object classes for both YOLO and Clothing detection',
            backstory=dedent("""
                You are an expert at understanding natural language queries and mapping them to 
                specific object detection classes. You have extensive knowledge of object categories 
                and can understand various ways users might refer to objects.
            """),
            verbose=True
        )

    def parse_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Parse a natural language query into a list of YOLO class names using CrewAI agents.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (valid_yolo_classes, valid_clothing_classes)
        """
        try:
            logger.info(f"Starting query parsing for: {query}")
            
            # Create the analysis task
            analysis_task = Task(
                description=dedent(f"""
                    Analyze the following query and identify and attempt to extract the classes from both YOLO and clothing categories.
                    Query: "{query}"
                    
                    Available YOLO classes: {self.yolo_classes}
                    Available Clothing classes: {self.clothing_classes}
                    
                    Expected Output:
                        A JSON object that matches the following structure (AnalyzedQueryOutput):
                        {{
                            "query": "the original query string",
                            "yolo_classes": ["list", "of", "valid", "yolo", "classes"],
                            "clothing_classes": ["list", "of", "valid", "clothing", "classes"]
                        }}
                    
                    Rules:
                        1. The query does not need exact word matching for classes. Extract only the most relevant classes from the query. 
                        2. The query can and will contain many words that are not classes. Do your best to only extract the classes the user wants to see. 
                        3. Your results should match the class names exactly.
                        4. Return the output as a valid JSON string.
                        5. If no classes are found for a category, return an empty list for that category.

                    Example Output for "Show me all vehicles and animals":
                        {{
                            "query": "Show me all vehicles and animals",
                            "yolo_classes": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
                            "clothing_classes": []
                        }}
                    
                    Example Output for "Show me all horses and cows":
                        {{
                            "query": "Show me all horses and cows",
                            "yolo_classes": ["horse", "cow"],
                            "clothing_classes": []
                        }}
                    
                    Example Output for "Show me all vehicles and people wearing shorts":
                        {{
                            "query": "Show me all vehicles and people wearing shorts",
                            "yolo_classes": ["car", "motorcycle", "bus", "truck", "person"],
                            "clothing_classes": ["Shorts_black-white", "Shorts_deep", "Shorts_pastel", "Shorts_pattern", "Shorts_vivid"]
                        }} 
                """),
                agent=self.query_analyzer,
                expected_output="A JSON object containing the query, a list of yolo_classes, and a list of clothing_classes",
                output_json=AnalyzedQueryOutput
            )

            # Create and run the crew
            crew = Crew(
                agents=[self.query_analyzer],
                tasks=[analysis_task],
                verbose=True
            )

            # Get the result
            result = crew.kickoff()
            logger.info(f"Crew result: {result}")
            
            # Extract classes directly from the result
            yolo_classes = result['yolo_classes'] or []
            clothing_classes = result['clothing_classes'] or []
            
            logger.info(f"Extracted YOLO classes: {yolo_classes}")
            logger.info(f"Extracted Clothing classes: {clothing_classes}")
            
            # Validate the classes
            valid_yolo_classes = self._validate_classes(yolo_classes, "yolo")
            valid_clothing_classes = self._validate_classes(clothing_classes, "clothing")
            
            return valid_yolo_classes, valid_clothing_classes
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return [], []

    def _validate_classes(self, classes: List[str], class_type: str) -> List[str]:
        """
        Validate classes against the appropriate class list.
        
        Args:
            classes: List of classes to validate
            class_type: Type of classes ("yolo" or "clothing")
            
        Returns:
            List of valid classes
        """
        try:
            available_classes = self.yolo_classes if class_type == "yolo" else self.clothing_classes
            valid_classes = []
            
            for class_name in classes:
                if class_name in available_classes:
                    valid_classes.append(class_name)
                    logger.info(f"Class '{class_name}' is valid for {class_type}")
                else:
                    logger.warning(f"Class '{class_name}' is not valid for {class_type}")
            
            return valid_classes
            
        except Exception as e:
            logger.error(f"Error validating classes: {e}")
            return []

    def get_available_classes(self) -> Tuple[List[str], List[str]]:
        """
        Get lists of available classes for both YOLO and clothing.
        
        Returns:
            Tuple of (yolo_classes, clothing_classes)
        """
        return self.yolo_classes, self.clothing_classes

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set
    if not OPENAI_API_KEY:
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
        
    language_service = LanguageService()
    query = "I would like to see all vehicles and people wearing shirts in the image."
    yolo_classes, clothing_classes = language_service.parse_query(query)
    print(f"\nQuery: {query}")
    print(f"Detected YOLO classes: {yolo_classes}")
    print(f"Detected Clothing classes: {clothing_classes}")