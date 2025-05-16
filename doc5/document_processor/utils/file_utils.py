"""
Utility functions for file operations
"""
import os
import uuid
import logging
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_EXTENSIONS = [
    'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt',
    'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'
]

def get_file_extension(file_path):
    """
    Get the file extension from a file path
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension without the dot
    """
    return os.path.splitext(file_path)[1][1:].lower()

def is_valid_document(file_path):
    """
    Check if the file is a supported document type
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if the file is a supported document type, False otherwise
    """
    return get_file_extension(file_path) in SUPPORTED_DOCUMENT_EXTENSIONS

def create_unique_filename(original_filename, prefix=''):
    """
    Create a unique filename to prevent overwrites
    
    Args:
        original_filename (str): Original filename
        prefix (str, optional): Prefix to add to the filename
        
    Returns:
        str: Unique filename
    """
    ext = get_file_extension(original_filename)
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    unique_id = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{base_name}_{unique_id}.{ext}"
    
    return f"{base_name}_{unique_id}.{ext}"

def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, create it if it doesn't
    
    Args:
        directory_path (str): Directory path
        
    Returns:
        str: Directory path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path

def convert_pdf_to_images(pdf_path, output_dir=None, dpi=300, first_page_only=False):
    """
    Convert a PDF file to a series of images
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the images. If None, uses the same directory as the PDF.
        dpi (int): DPI for the output images
        first_page_only (bool): If True, only convert the first page
        
    Returns:
        list: List of paths to the generated images
    """
    try:
        # Ensure output directory exists
        if output_dir:
            output_dir = ensure_directory_exists(output_dir)
        else:
            output_dir = os.path.dirname(pdf_path)
            
        # Create base name for output images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # List to store image paths
        image_paths = []
        
        # Determine the pages to convert
        pages_to_convert = [0] if first_page_only else range(len(pdf_document))
        
        # Convert each page
        for page_num in pages_to_convert:
            page = pdf_document[page_num]
            
            # Get the pixmap (image) of the page
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Create output image filename
            img_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1}.png")
            
            # Save image
            pix.save(img_path)
            image_paths.append(img_path)
            
            logger.info(f"Converted page {page_num+1} of {pdf_path} to {img_path}")
        
        pdf_document.close()
        return image_paths
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        return []

def get_image_dimensions(image_path):
    """
    Get the dimensions of an image
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        tuple: (width, height) of the image
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error getting image dimensions: {str(e)}")
        return (0, 0)  # Default fallback dimensions
