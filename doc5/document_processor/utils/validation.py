"""
Input validation utilities for document processing
"""
import os
import logging
from typing import List, Tuple, Optional, Dict, Any

from document_processor.utils.custom_exceptions import (
    FileTypeError, FileSizeError, ValidationError, FileReadError
)

logger = logging.getLogger(__name__)

def validate_file(file_path: str, 
                 allowed_extensions: Optional[List[str]] = None, 
                 max_size_bytes: Optional[int] = None) -> Tuple[bool, Optional[Exception]]:
    """
    Validate a file against size and type constraints
    
    Args:
        file_path (str): Path to the file
        allowed_extensions (List[str], optional): List of allowed file extensions
        max_size_bytes (int, optional): Maximum file size in bytes
        
    Returns:
        Tuple[bool, Optional[Exception]]: Tuple containing validation result and exception if failed
    """
    # Default allowed extensions if not specified
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']
    
    # Ensure all extensions are lowercase for case-insensitive comparison
    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    
    # Default max size if not specified (10 MB)
    if max_size_bytes is None:
        max_size_bytes = 10 * 1024 * 1024
    
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False, FileReadError(file_path, "File does not exist")
    
    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.debug(f"Validating file extension: {file_extension} against allowed: {allowed_extensions}")
    if allowed_extensions and file_extension not in allowed_extensions:
        logger.error(f"Unsupported file type: {file_path} with extension {file_extension}")
        return False, FileTypeError(file_path, allowed_extensions)
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if max_size_bytes and file_size > max_size_bytes:
        logger.error(f"File size exceeds maximum allowed: {file_size} bytes")
        return False, FileSizeError(file_path, file_size, max_size_bytes)
    
    logger.debug(f"File validation successful for {file_path}")
    return True, None

def validate_text(text: str, min_length: int = 10) -> Tuple[bool, Optional[Exception]]:
    """
    Validate text content
    
    Args:
        text (str): Text to validate
        min_length (int, optional): Minimum text length
        
    Returns:
        Tuple[bool, Optional[Exception]]: Tuple containing validation result and exception if failed
    """
    if not text:
        logger.error("Text is empty")
        return False, ValidationError("text", "Text content is empty")
    
    if len(text.strip()) < min_length:
        logger.error(f"Text length ({len(text.strip())}) is less than minimum required ({min_length})")
        return False, ValidationError("text", f"Text content is too short (minimum: {min_length} characters)", len(text.strip()))
    
    return True, None

def validate_json_request(request_data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, Optional[Exception]]:
    """
    Validate JSON request data
    
    Args:
        request_data (Dict[str, Any]): Request data to validate
        required_fields (List[str]): List of required fields
        
    Returns:
        Tuple[bool, Optional[Exception]]: Tuple containing validation result and exception if failed
    """
    if not request_data:
        logger.error("Empty request data")
        return False, ValidationError("request", "Request data is empty")
    
    for field in required_fields:
        if field not in request_data:
            logger.error(f"Missing required field: {field}")
            return False, ValidationError(field, "Required field is missing")
        
        if request_data[field] is None or (isinstance(request_data[field], str) and not request_data[field].strip()):
            logger.error(f"Empty required field: {field}")
            return False, ValidationError(field, "Field cannot be empty")
    
    return True, None

def validate_session_id(session_id: str) -> Tuple[bool, Optional[Exception]]:
    """
    Validate session ID format
    
    Args:
        session_id (str): Session ID to validate
        
    Returns:
        Tuple[bool, Optional[Exception]]: Tuple containing validation result and exception if failed
    """
    import re
    
    if not session_id:
        logger.error("Empty session ID")
        return False, ValidationError("session_id", "Session ID is empty")
    
    # Validate UUID format (simple regex for UUID v4)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)
    if not uuid_pattern.match(session_id):
        logger.error(f"Invalid session ID format: {session_id}")
        return False, ValidationError("session_id", "Invalid session ID format", session_id)
    
    return True, None