from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from language_service import LanguageService

import logging

logger = logging.getLogger(__name__)

language_router = APIRouter(
    prefix="/language",
    tags=["language"],
)

language_service = LanguageService()

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    yolo_classes: list[str]
    clothing_classes: list[str]
    
@language_router.post("/parse_query", response_model=QueryResponse)
async def parse_query(request: QueryRequest):
    """
    Parse a natural language query into both YOLO and Clothing classes.
    
    Args:
        request: QueryRequest containing the natural language query
        
    Returns:
        QueryResponse with parsed YOLO and Clothing classes
    """
    try:
        logger.info(f"Parsing query: {request.query}")
        
        yolo_classes, clothing_classes = language_service.parse_query(request.query)
        
        logger.info(f"Parsed YOLO classes: {yolo_classes}")
        logger.info(f"Parsed Clothing classes: {clothing_classes}")
        
        return QueryResponse(
            yolo_classes=yolo_classes,
            clothing_classes=clothing_classes
        )
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

