import logging
import os
from typing import List
from crewai import Agent, Task, Crew
from textwrap import dedent
from dotenv import load_dotenv
from pydantic import BaseModel

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
    classes: List[str]

class LanguageService():
    def __init__(self):
        # Define the YOLO classes
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

        # Create the query analyzer agent
        self.query_analyzer = Agent(
            role='Query Analyzer',
            goal='Analyze user queries and identify relevant YOLO object classes',
            backstory=dedent("""
                You are an expert at understanding natural language queries and mapping them to 
                specific object detection classes. You have extensive knowledge of object categories 
                and can understand various ways users might refer to objects.
            """),
            verbose=True
        )

        # Create the class validator agent
        self.class_validator = Agent(
            role='Class Validator',
            goal='Validate and refine the identified classes',
            backstory=dedent("""
                You are an expert at validating object detection classes. You ensure that only 
                valid YOLO classes are returned and that the results make sense in the context 
                of the user's query.
            """),
            verbose=True
        )

    def parse_query(self, query: str) -> List[str]:
        """
        Parse a natural language query into a list of YOLO class names using CrewAI agents.
        
        Args:
            query: The user's natural language query
            
        Returns:
            List of YOLO class names to detect
        """
        try:
            logger.info(f"Starting query parsing for: {query}")
            
            # Create the analysis task
            analysis_task = Task(
                description=dedent(f"""
                    Analyze the following query and identify and attempt to extract the classes from this set of yolo classes {self.yolo_classes}
                    Query: "{query}"
                    
                    Expected Output:
                        A JSON string that matches the following structure:
                        {{
                            "query": "the original query string",
                            "classes": ["list", "of", "valid", "yolo", "classes"]
                        }}
                    
                    Rules:
                        1. The query does not need exact word matching for classes. Extract only the most relevant classes from the query. 
                        2. The query can and will contain many words that are not classes. Do your best to only extract the classes the user wants to see. 
                        3. Your results should match the class names exactly.
                        4. Return the output as a valid JSON string.

                    Example Output for "Show me all vehicles and animals":
                        {{
                            "query": "Show me all vehicles and animals",
                            "classes": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
                        }}
                    
                    Example Output for "Show me all horses and cows":
                        {{
                            "query": "Show me all horses and cows",
                            "classes": ["horse", "cow"]
                        }}
                """),
                agent=self.query_analyzer,
                expected_output="A JSON string containing the query and list of classes"
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
            
            # Parse the result to extract class names
            detected_classes = self._extract_classes_from_result(result)
            logger.info(f"Extracted classes: {detected_classes}")
            
            if not detected_classes:
                logger.warning(f"No matching classes found for query: {query}")
                return []
            
            logger.info(f"Parsed query '{query}' into classes: {detected_classes}")
            return detected_classes
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return []

    def _validate_classes(self, results: List[str]) -> List[str]:
        """
        Extract class names from the CrewAI result.
        
        Args:
            results: The list of classes from the CrewAI result
            
        Returns:
            List of valid YOLO class names
        """
        try:
            valid_classes = []
            for class_name in results:
                if class_name in self.yolo_classes:
                    valid_classes.append(class_name)
                    logger.info(f"Class '{class_name}' is a valid YOLO class")
                else:
                    logger.warning(f"Class '{class_name}' is not a valid YOLO class")

            return valid_classes
            
        except Exception as e:
            logger.error(f"Error extracting classes from result: {e}")
            return []

    def get_available_classes(self) -> List[str]:
        """Get list of all available YOLO classes."""
        return self.yolo_classes

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set
    if not OPENAI_API_KEY:
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
        
    language_service = LanguageService()
    query = "I would like to see all vehicles and horses in the image."
    classes = language_service.parse_query(query)
    print(f"\nQuery: {query}")
    print(f"Detected classes: {classes}")